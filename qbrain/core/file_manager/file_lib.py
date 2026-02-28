
import base64
import json
import os
import random
import tempfile
from datetime import datetime
from typing import Dict, Any, List, TypedDict, Callable, Optional

import dotenv
from qbrain.core.file_manager.extractor import RawModuleExtractor
from qbrain.core.qbrain_manager import get_qbrain_table_manager
from qbrain.core.handler_utils import require_param, require_param_truthy, get_val
from qbrain.a_b_c.bq_agent._bq_core.bq_handler import BQCore
dotenv.load_dotenv()

# Optional Vertex RAG integration (upsert at receive, retrieval during extraction)
def _vertex_rag_available() -> bool:
    try:
        from vertex_rag.config import get_default_config
        cfg = get_default_config()
        return bool(cfg.rag_corpus_id)
    except Exception:
        return False


def _upsert_files_to_vertex_rag(
    module_id: str,
    file_bytes_list: List[bytes],
    user_id: str,
    rag_corpus_id: Optional[str] = None,
) -> List[str]:
    """
    Write each file's bytes to a temp file, upload to Vertex RAG corpus, return RAG file names (ids).
    Returns empty list if RAG is disabled or upload fails.
    """
    if not _vertex_rag_available() or not file_bytes_list:
        return []
    try:
        from vertex_rag.upsert import upload_local_file
        from vertex_rag.config import get_default_config, VertexRagConfig
        rag_file_ids: List[str] = []
        cfg: VertexRagConfig = get_default_config()
        if rag_corpus_id:
            cfg.rag_corpus_id = rag_corpus_id
        for i, b in enumerate(file_bytes_list):
            suffix = ".pdf"  # RAG supports PDF; keep default for other types if needed
            try:
                import filetype
                kind = filetype.guess(b)
                if kind and kind.extension:
                    suffix = f".{kind.extension}"
            except Exception:
                pass
            fd, path = tempfile.mkstemp(suffix=suffix, prefix=f"fm_{module_id}_")
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(b)
                display_name = f"{module_id}_f{i}"
                info = upload_local_file(
                    path=path,
                    display_name=display_name,
                    description=f"user={user_id} module={module_id}",
                    config=cfg,
                )
                name = info.get("name")
                if name:
                    rag_file_ids.append(name)
                    print(f"[FileManager] Vertex RAG upserted file -> {name}")
            finally:
                try:
                    os.unlink(path)
                except Exception:
                    pass
        print(f"[FileManager] Vertex RAG: {len(rag_file_ids)} file(s) upserted")
        return rag_file_ids
    except Exception as e:
        print(f"[FileManager] Vertex RAG upsert skipped: {e}")
        return []


def _rag_retrieval_context(
    query: str,
    rag_file_ids: List[str],
    top_k: int = 10,
    rag_corpus_id: Optional[str] = None,
) -> str:
    """
    Run Vertex RAG retrieval for the given query scoped to rag_file_ids; return formatted context string.
    """
    if not _vertex_rag_available() or not rag_file_ids or not query.strip():
        return ""
    try:
        from vertex_rag.retrieval import retrieval_query
        from vertex_rag.config import get_default_config, VertexRagConfig
        cfg: VertexRagConfig = get_default_config()
        if rag_corpus_id:
            cfg.rag_corpus_id = rag_corpus_id

        result = retrieval_query(
            text=query,
            top_k=top_k,
            rag_file_ids=rag_file_ids,
            config=cfg,
        )
        contexts = result.get("contexts") or []
        parts: List[str] = []
        for ctx in contexts:
            chunk = ctx.get("chunk")
            if chunk is None:
                continue
            if hasattr(chunk, "content"):
                parts.append(getattr(chunk, "content", "") or "")
            elif hasattr(chunk, "text"):
                parts.append(getattr(chunk, "text", "") or "")
            elif isinstance(chunk, dict):
                parts.append(chunk.get("content") or chunk.get("text") or "")
            else:
                parts.append(str(chunk))
        return "\n\n".join(p for p in parts if p.strip())
    except Exception as e:
        print(f"[FileManager] Vertex RAG retrieval skipped: {e}")
        return ""


class EquationContent(TypedDict, total=False):
    """Struct for extracted math/LaTeX from PDF preprocessing."""
    latex: List[str]
    math_elements: List[str]
    equations: List[str]


def _files_to_bytes(files: List) -> List[bytes]:
    """Convert file payloads (base64 strings, bytes, or file-like) to bytes list."""
    result = []
    for f in files or []:
        try:
            if isinstance(f, bytes):
                result.append(f)
            elif isinstance(f, str):
                s = f.split("base64,")[-1] if "base64," in f else f
                result.append(base64.b64decode(s))
            elif hasattr(f, "read"):
                result.append(f.read())
            else:
                continue
        except Exception as e:
            print(f"[FileManager] files_to_bytes skip item: {e}")
    print(f"[FileManager] files_to_bytes: converted {len(result)} file(s) to bytes")
    return result


class FileManager(RawModuleExtractor):
    DATASET_ID = "QBRAIN"
    MODULES_TABLE = "modules"
    FILES_TABLE = "files"

    def __init__(self, qb):
        RawModuleExtractor.__init__(self)
        self.qb = qb

    file_params: dict = {}  # Testing only attribute

    def _req_struct_to_json_schema(self, req_struct: dict, content_type: str, data_key: str) -> dict:
        """
        Convert req_struct from SET case to JSON Schema for Gemini.
        Uses the exact req_struct as the Tool schema for the output.
        """
        data_struct = req_struct.get("data", req_struct) or {}
        item_def = data_struct.get(data_key, data_struct) if isinstance(data_struct, dict) else data_struct

        def dict_to_schema(d: dict) -> dict:
            props = {}
            for k, v in d.items():
                desc = str(v) if isinstance(v, str) else ""
                if isinstance(v, str) and "list" in v:
                    props[k] = {"type": "array", "items": {"type": "string"}, "description": desc}
                elif isinstance(v, str) and "dict" in v:
                    props[k] = {"type": "object", "additionalProperties": True, "description": desc}
                elif isinstance(v, dict):
                    props[k] = dict_to_schema(v)
                else:
                    props[k] = {"type": "string", "description": desc}
            return {"type": "object", "properties": props, "additionalProperties": True}

        if isinstance(item_def, dict) and item_def:
            item_schema = dict_to_schema(item_def)
            item_schema["description"] = f"Single item for SET_{content_type.upper()} handler"
            req_keys = {"param": ["name"], "field": ["name"], "method": ["equation"]}
            if content_type in req_keys:
                item_schema["required"] = [r for r in req_keys[content_type] if r in (item_schema.get("properties") or {})]
        else:
            # req_struct has string type hint (e.g. param: "dict|list"); use explicit schema from case
            item_schema = {
                "type": "object",
                "description": f"Single item for SET_{content_type.upper()} handler",
                "additionalProperties": True,
            }

        return {
            "type": "object",
            "description": f"Output matching SET_{content_type.upper()} req_struct",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Extracted items for handler",
                    "items": item_schema,
                },
            },
            "required": ["items"],
        }

    def _schema_to_genai(self, schema: dict) -> dict:
        """Convert JSON Schema type values to GenAI uppercase (OBJECT, ARRAY, STRING, NUMBER)."""
        if not isinstance(schema, dict):
            return schema
        out = {}
        for k, v in schema.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.upper()
            elif k == "properties" and isinstance(v, dict):
                out[k] = {pk: self._schema_to_genai(pv) for pk, pv in v.items()}
            elif k == "items" and isinstance(v, dict):
                out[k] = self._schema_to_genai(v)
            else:
                out[k] = self._schema_to_genai(v) if isinstance(v, dict) else v
        return out

    def _extract_component_with_rag(
        self,
        rag_corpus_id: str,
        rag_file_ids: List[str],
        content_type: str,
        user_prompt: str,
        session_id: Optional[str] = None,
        params_list: Optional[List[Dict[str, Any]]] = None,
        fallback_users_params: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run one component extraction (param, field, method) via Gem.ask_rag with
        session corpus, out_struct, and the given rag_file_ids (e.g. from orchestrator.last_files).
        """
        if not rag_file_ids or not rag_corpus_id:
            return []
        try:
            from vertex_rag.config import get_corpus_name
            from gem_core.gem import Gem
        except Exception as e:
            print(f"[FileManager] _extract_component_with_rag: import error: {e}")
            return []
        corpus_name = get_corpus_name(rag_corpus_id)
        if not corpus_name:
            print("[FileManager] _extract_component_with_rag: no corpus_name for rag_corpus_id")
            return []

        case_map = {
            "param": ("core.param_manager.case", "RELAY_PARAM", "SET_PARAM", "param"),
            "field": ("core.fields_manager.case", "RELAY_FIELD", "SET_FIELD", "field"),
            "method": ("core.method_manager.case", "RELAY_METHOD", "SET_METHOD", "data"),
        }

        if content_type not in case_map:
            return []

        # define reponse scheme
        mod_name, attr, case_name, data_key = case_map[content_type]
        mod = __import__(mod_name, fromlist=[attr])
        relay = getattr(mod, attr, [])
        set_case = next((c for c in relay if c.get("case") == case_name), None)
        req_struct = set_case.get("req_struct", {}) if set_case else {}
        json_schema = self._req_struct_to_json_schema(req_struct, content_type, data_key)
        response_schema = self._schema_to_genai(json_schema)

        prompts = {
            "param": "From the retrieved document context, extract all parameters (scalars, vectors, matrices, indices) and return them as a JSON object with an 'items' array. Each item must have name, type, value where applicable. Exclude parameters that already exist in the user's params. User instructions: ",
            "field": "From the retrieved document context, extract all fields (physical/quantum fields) and return them as a JSON object with an 'items' array. Use the available params for field definitions. User instructions: ",
            "method": "From the retrieved document context, extract all methods/equations and return them as a JSON object with an 'items' array. Each item must have equation, description, params. User instructions: ",
        }

        base_prompt = prompts.get(content_type, "Extract relevant items from the retrieved context. User instructions: ")
        prompt = f"{base_prompt}{user_prompt or 'Extract all relevant content of this type.'}"
        if content_type == "param" and fallback_users_params:
            prompt += f"\n\nExisting user params (exclude duplicates): {json.dumps(fallback_users_params[:30], default=str)}"
        if content_type in ("field", "method") and params_list:
            prompt += f"\n\nAvailable params: {json.dumps([p.get('name') or p.get('id') for p in params_list[:50]], default=str)}"

        try:
            gem = Gem()
            raw = gem.ask_rag(
                prompt=prompt,
                corpus_name=corpus_name,
                rag_file_ids=rag_file_ids,
                top_k=10,
                response_schema=response_schema,
            )
            text = (raw or "").strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            items = parsed.get("items", parsed) if isinstance(parsed, dict) else parsed
            result = items if isinstance(items, list) else [items]
            print(f"[FileManager] _extract_component_with_rag: {content_type} -> {len(result)} item(s)")
            return [r for r in result if isinstance(r, dict)]
        except Exception as e:
            print(f"[FileManager] _extract_component_with_rag {content_type} error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _extract_with_struct(
        self,
        equation_content: EquationContent,
        user_prompt: str,
        content_type: str,
        pdf_bytes: bytes = None,
    ) -> List[Dict[str, Any]]:
        """
        Extract items using the exact req_struct of the SET method as Tool (JSON schema) for Gemini.
        Inputs: file bytes (PDF) + user prompt. Sys instructions guide generation for the specific handler.
        """
        print(f"[FileManager] _extract_with_struct: starting content_type={content_type}")
        from gem_core.gem import Gem
        case_map = {
            "param": ("core.param_manager.case", "RELAY_PARAM", "SET_PARAM", "param"),
            "field": ("core.fields_manager.case", "RELAY_FIELD", "SET_FIELD", "field"),
            "method": ("core.method_manager.case", "RELAY_METHOD", "SET_METHOD", "data"),
        }
        mod_name, attr, case_name, data_key = case_map[content_type]
        mod = __import__(mod_name, fromlist=[attr])
        relay = getattr(mod, attr, [])
        set_case = next((c for c in relay if c.get("case") == case_name), None)
        req_struct = set_case.get("req_struct", {}) if set_case else {}
        json_schema = self._req_struct_to_json_schema(req_struct, content_type, data_key)
        print(f"[FileManager] _extract_with_struct: loaded req_struct for {content_type}, schema keys: {list(json_schema.get('properties', {}).keys())}")

        # System instructions: generate data for the specific SET method
        set_name = f"SET_{content_type.upper()}"
        sys_instructions = f"""You are extracting structured data from a scientific document for the {set_name} handler.
Your output must conform exactly to the provided JSON schema (Tool), which matches the handler's req_struct.

User instructions: {user_prompt or 'Extract all relevant content of this type.'}

Pre-extracted equation content from the document (use as context; the PDF is also attached):
- latex: {json.dumps(equation_content.get('latex', [])[:50], default=str)}
- math_elements: {json.dumps(equation_content.get('math_elements', [])[:50], default=str)}
- equations: {json.dumps(equation_content.get('equations', [])[:20], default=str)}

Generate a JSON object with an "items" array. Each item must match the schema exactly.
Return only valid JSON, no markdown or extra text."""

        config = {
            "response_mime_type": "application/json",
            "response_json_schema": json_schema,
        }

        try:
            gem = Gem()
            if pdf_bytes:
                try:
                    print(f"[FileManager] _extract_with_struct: calling Gemini (multimodal: PDF + schema Tool)")
                    b64 = base64.b64encode(pdf_bytes).decode("ascii")
                    response = gem.ask_mm(
                        file_content_str=f"data:application/pdf;base64,{b64}",
                        prompt=sys_instructions,
                        config=config,
                    )
                except Exception as e1:
                    print(f"[FileManager] _extract_with_struct: multimodal failed ({e1}), falling back to text-only")
                    response = gem.ask(sys_instructions, config=config)
            else:
                print(f"[FileManager] _extract_with_struct: calling Gemini (text-only + schema Tool)")
                response = gem.ask(sys_instructions, config=config)
            text = (response or "").strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(text)
            items = parsed.get("items", parsed) if isinstance(parsed, dict) else parsed
            result = items if isinstance(items, list) else [items]
            print(f"[FileManager] _extract_with_struct: {content_type} done -> {len(result)} item(s)")
            return result
        except Exception as e:
            print(f"[FileManager] _extract_with_struct {content_type} error: {e}")
            return []

    # -------------------------------------------------------------
    # WORKFLOW STEP HELPERS (CALLED FROM process_and_upload_file_config)
    # -------------------------------------------------------------

    def _step0_prepare_inputs(
        self,
        user_id: str,
        data: Dict[str, Any],
        last_files: Optional[List[str]],
        rag_corpus_id: Optional[str],
        mock_extraction: bool,
    ) -> tuple[str, List[bytes], str]:
        """
        STEP 0: PREPARE INPUTS (MODULE ID, FILE BYTES, USER PROMPT)
        """
        module_id = data.get("id") or f"file_{random.randint(100000, 999999)}"
        files_raw = data.get("files", [])
        file_bytes_list = _files_to_bytes(files_raw)
        user_prompt = data.get("prompt", "") or data.get("msg", "")
        print(
            f"[FileManager] process_and_upload_file_config: "
            f"module_id={module_id}, files={len(file_bytes_list)}, "
            f"last_files={len(last_files or [])}, rag_corpus_id={bool(rag_corpus_id)}, "
            f"mock_extraction={mock_extraction}"
        )
        return module_id, file_bytes_list, user_prompt

    def _step1_vertex_rag_upsert(
        self,
        module_id: str,
        file_bytes_list: List[bytes],
        user_id: str,
        rag_corpus_id: Optional[str],
        testing: bool,
    ) -> List[str]:
        """
        STEP 1: VERTEX RAG UPSERT (FILES -> RAG_FILE_IDS)
        """
        rag_file_ids: List[str] = []
        if not testing and file_bytes_list:
            rag_file_ids = _upsert_files_to_vertex_rag(
                module_id, file_bytes_list, user_id, rag_corpus_id=rag_corpus_id
            )
        print(
            f"[FileManager] process_and_upload_file_config: rag_file_ids={len(rag_file_ids)}"
        )
        return rag_file_ids

    def _step2_extract_components_pipeline(
        self,
        user_id: str,
        file_bytes_list: List[bytes],
        user_prompt: str,
        rag_corpus_id: Optional[str],
        rag_file_ids: List[str],
        session_id: Optional[str],
        last_files: Optional[List[str]],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        STEP 2: COMPONENT EXTRACTION (PARAM, FIELD, METHOD)
        """
        # RAG file list for extraction: current upload ids, or orchestrator's last_files
        rag_file_ids_for_rag: List[str] = rag_file_ids if rag_file_ids else (last_files or [])

        extracted_ids: Dict[str, List[str]] = {"param": [], "field": [], "method": []}
        # Fetched here for both RAG-based and fallback extraction
        from qbrain.core.managers_context import get_param_manager
        fallback_users_params = get_param_manager().get_users_params(user_id)

        use_rag_extraction = bool(rag_corpus_id and rag_file_ids_for_rag)

        if use_rag_extraction:
            # STEP 2A: GEM.ASK_RAG EXTRACTION (USING OUT_STRUCT + SESSION CORPUS + LAST_FILES)
            params_list = self._extract_component_with_rag(
                rag_corpus_id=rag_corpus_id,
                rag_file_ids=rag_file_ids_for_rag,
                content_type="param",
                user_prompt=user_prompt,
                session_id=session_id,
                fallback_users_params=fallback_users_params,
            )
            methods_list = self._extract_component_with_rag(
                rag_corpus_id=rag_corpus_id,
                rag_file_ids=rag_file_ids_for_rag,
                content_type="method",
                user_prompt=user_prompt,
                session_id=session_id,
                params_list=params_list,
                fallback_users_params=fallback_users_params,
            )
            fields_list = self._extract_component_with_rag(
                rag_corpus_id=rag_corpus_id,
                rag_file_ids=rag_file_ids_for_rag,
                content_type="field",
                user_prompt=user_prompt,
                session_id=session_id,
                params_list=params_list,
                fallback_users_params=fallback_users_params,
            )
        else:
            # STEP 2B: FALLBACK EXTRACTION (MANAGER extract_from_file_bytes WITH OPTIONAL RAG CONTEXT)
            files_b64 = [base64.b64encode(b).decode("ascii") for b in file_bytes_list]
            equation_content = "\n".join(files_b64)
            extraction_content = equation_content
            if rag_file_ids:
                rag_query = (
                    user_prompt
                    or "parameters, fields, methods, and equations from the document"
                ).strip()
                rag_context = _rag_retrieval_context(
                    rag_query, rag_file_ids, top_k=10, rag_corpus_id=rag_corpus_id
                )
                if rag_context:
                    extraction_content = (
                        f"[RAG context]\n{rag_context}\n\n[Document content]\n{equation_content}"
                    )

            from qbrain.core.managers_context import get_param_manager
            params_raw = get_param_manager().extract_from_file_bytes(
                content=extraction_content,
                instructions=user_prompt,
                users_params=fallback_users_params,
            )
            params_list: List[Dict[str, Any]] = []
            if isinstance(params_raw, dict):
                _p = params_raw.get("param") or params_raw.get("params")
                if isinstance(_p, list):
                    params_list = [p for p in _p if isinstance(p, dict)]
                elif isinstance(_p, dict):
                    params_list = [_p]
            elif isinstance(params_raw, list):
                params_list = [p for p in params_raw if isinstance(p, dict)]

            from qbrain.core.managers_context import get_method_manager
            methods_raw = get_method_manager().extract_from_file_bytes(
                content=extraction_content,
                instructions=user_prompt,
                params=params_list,
                fallback_params=fallback_users_params,
            )
            methods_list: List[Dict[str, Any]] = []
            if isinstance(methods_raw, dict):
                _m = methods_raw.get("methods")
                if isinstance(_m, list):
                    methods_list = [m for m in _m if isinstance(m, dict)]
                elif isinstance(_m, dict):
                    methods_list = [_m]
            elif isinstance(methods_raw, list):
                methods_list = [m for m in methods_raw if isinstance(m, dict)]

            from qbrain.core.managers_context import get_field_manager
            fields_raw = get_field_manager().extract_from_file_bytes(
                extraction_content,
                params_list,
                user_prompt,
                fallback_users_params,
            )
            fields_list: List[Dict[str, Any]] = []
            if isinstance(fields_raw, dict):
                _f = fields_raw.get("field") or fields_raw.get("fields")
                if isinstance(_f, list):
                    fields_list = [f for f in _f if isinstance(f, dict)]
                elif isinstance(_f, dict):
                    fields_list = [_f]
            elif isinstance(fields_raw, list):
                fields_list = [f for f in fields_raw if isinstance(f, dict)]

        return params_list, fields_list, methods_list

    def _step3_upsert_components(
        self,
        user_id: str,
        params_list: List[Dict[str, Any]],
        fields_list: List[Dict[str, Any]],
        methods_list: List[Dict[str, Any]],
        testing: bool,
    ) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
        """
        STEP 3: UPSERT COMPONENTS (PARAM, FIELD, METHOD TABLES)
        """
        from qbrain.core.managers_context import get_param_manager, get_field_manager, get_method_manager
        extracted_ids: Dict[str, List[str]] = {"param": [], "field": [], "method": []}
        created_components: Dict[str, Any] = {"param": [], "field": [], "method": []}

        # PARAMS
        if params_list:
            ids = []
            for item in params_list:
                pid = item.get("id") or item.get("name") or str(
                    random.randint(100000, 999999)
                )
                item["id"] = pid
                item["name"] = item.get("name") or pid
                ids.append(pid)
            if not testing:
                get_param_manager().set_param(params_list, user_id)
            extracted_ids["param"].extend(ids)
            created_components["param"] = [
                {"auth": {"user_id": user_id}, "data": {"param": params_list}}
            ]
            print(
                f"[FileManager] process_and_upload_file_config: params upserted -> {ids}"
            )

        # FIELDS
        if fields_list:
            ids = []
            for item in fields_list:
                fid = item.get("id") or str(random.randint(100000, 999999))
                item["id"] = fid
                ids.append(fid)
            if not testing:
                get_field_manager().set_field(fields_list, user_id)
            extracted_ids["field"].extend(ids)
            created_components["field"] = [
                {"auth": {"user_id": user_id}, "data": {"field": f}}
                for f in fields_list
            ]
            print(
                f"[FileManager] process_and_upload_file_config: fields upserted -> {ids}"
            )

        # METHODS
        if methods_list:
            ids = []
            for m in methods_list:
                mid = m.get("id") or str(random.randint(100000, 999999))
                m["id"] = mid
                m["user_id"] = user_id
                if "equation" in m and "code" not in m:
                    m["code"] = m["equation"]
                ids.append(mid)
            if not testing:
                get_method_manager().set_method(methods_list, user_id)
            extracted_ids["method"].extend(ids)
            created_components["method"] = [
                {"auth": {"user_id": user_id}, "data": dict(m)} for m in methods_list
            ]
            print(
                f"[FileManager] process_and_upload_file_config: methods upserted -> {ids}"
            )

        return extracted_ids, created_components

    def _step4_upsert_files_table(
        self,
        module_id: str,
        user_id: str,
        file_bytes_list: List[bytes],
        rag_file_ids: List[str],
        testing: bool,
    ) -> None:
        """
        STEP 4: UPSERT FILE ROWS (FILES TABLE)
        """
        if testing or not file_bytes_list:
            print(
                "[FileManager] process_and_upload_file_config: skipping files table (testing or no files)"
            )
            return

        for i, _ in enumerate(file_bytes_list):
            file_row = {
                "id": f"{module_id}_f{i}" if len(file_bytes_list) > 1 else module_id,
                "user_id": user_id,
                "module_id": module_id,
                "created_at": datetime.utcnow().isoformat(),
            }
            if i < len(rag_file_ids):
                file_row["rag_file_id"] = rag_file_ids[i]
            try:
                self.qb.set_item(self.FILES_TABLE, file_row)
                print(
                    f"[FileManager] process_and_upload_file_config: file row upserted -> {file_row.get('id')}"
                )
            except Exception as e:
                print(f"[FileManager] files table upsert skipped: {e}")

    def _step5_upsert_module(
        self,
        user_id: str,
        data: Dict[str, Any],
        module_id: str,
        rag_file_ids: List[str],
        testing: bool,
    ) -> None:
        """
        STEP 5: UPSERT MODULE ROW (MODULES TABLE)
        """
        if testing:
            return

        row = {**data, "user_id": user_id}
        row.pop("methods", None)
        row.pop("fields", None)
        row.pop("files", None)  # files are not JSON-serializable (handles/bytes)
        if rag_file_ids:
            row["rag_file_ids"] = rag_file_ids  # save local (RAG) file ids for retrieval
        self.set_module(row, user_id)
        print("[FileManager] process_and_upload_file_config: module upserted")

    def process_and_upload_file_config(
        self,
        user_id: str,
        data: Dict[str, Any],
        testing: bool = False,
        mock_extraction: bool = False,
        rag_corpus_id: Optional[str] = None,
        session_id: Optional[str] = None,
        last_files: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Extract case-specific content from files via param/field/method managers,
        upsert via their set methods, upsert file metadata to files table.
        When rag_corpus_id and (current rag_file_ids or last_files) are set, all component
        extraction uses Gem.ask_rag with session corpus, out_struct, and that file list.
        Returns type=CONTENT_EXTRACTED, data={...}, created_components, rag_file_ids.

        last_files: RAG file ids from orchestrator.last_files (used for ask_rag when no new upload).
        mock_extraction: If True, skip Gemini and use sample data (for quick tests).
        """
        print("[FileManager] process_and_upload_file_config: starting")

        # STEP 0: PREPARE INPUTS (MODULE ID, FILE BYTES, USER PROMPT)
        module_id, file_bytes_list, user_prompt = self._step0_prepare_inputs(
            user_id=user_id,
            data=data,
            last_files=last_files,
            rag_corpus_id=rag_corpus_id,
            mock_extraction=mock_extraction,
        )

        # STEP 1: VERTEX RAG UPSERT (FILES -> RAG_FILE_IDS)
        rag_file_ids = self._step1_vertex_rag_upsert(
            module_id=module_id,
            file_bytes_list=file_bytes_list,
            user_id=user_id,
            rag_corpus_id=rag_corpus_id,
            testing=testing,
        )

        # STEP 2: COMPONENT EXTRACTION (PARAM, FIELD, METHOD)
        params_list, fields_list, methods_list = self._step2_extract_components_pipeline(
            user_id=user_id,
            file_bytes_list=file_bytes_list,
            user_prompt=user_prompt,
            rag_corpus_id=rag_corpus_id,
            rag_file_ids=rag_file_ids,
            session_id=session_id,
            last_files=last_files,
        )

        # STEP 3: UPSERT COMPONENTS (PARAM, FIELD, METHOD TABLES)
        extracted_ids, created_components = self._step3_upsert_components(
            user_id=user_id,
            params_list=params_list,
            fields_list=fields_list,
            methods_list=methods_list,
            testing=testing,
        )

        # STEP 4: UPSERT FILE ROWS (FILES TABLE)
        self._step4_upsert_files_table(
            module_id=module_id,
            user_id=user_id,
            file_bytes_list=file_bytes_list,
            rag_file_ids=rag_file_ids,
            testing=testing,
        )

        # STEP 5: UPSERT MODULE ROW (MODULES TABLE)
        self._step5_upsert_module(
            user_id=user_id,
            data=data,
            module_id=module_id,
            rag_file_ids=rag_file_ids,
            testing=testing,
        )

        print(
            f"[FileManager] process_and_upload_file_config: done -> "
            f"param={len(extracted_ids['param'])}, "
            f"field={len(extracted_ids['field'])}, "
            f"method={len(extracted_ids['method'])}"
        )
        return {
            "type": "CONTENT_EXTRACTED",
            "data": {
                "param": list(dict.fromkeys(extracted_ids["param"])),
                "field": list(dict.fromkeys(extracted_ids["field"])),
                "method": list(dict.fromkeys(extracted_ids["method"])),
            },
            "created_components": created_components,
            "rag_file_ids": rag_file_ids,
        }

    def _classify_and_call(
        self,
        user_prompt: str,
        options: Dict[str, Callable[[], Any]],
        default_key: Optional[str] = None,
    ):
        """
        Classify the user prompt into one of the option keys and invoke the corresponding callable.

        options: dict mapping option key -> zero-arg callable.
        """
        if not options:
            raise ValueError("[FileManager] _classify_and_call: options dict is empty")

        keys = list(options.keys())
        chosen_key: Optional[str] = None

        # 1) Try LLM-based routing using Gem (short text-only prompt).
        try:
            from gem_core.gem import Gem

            gem = Gem()
            opt_list = ", ".join(keys)
            routing_prompt = (
                "You are a router deciding how to handle a file-related request.\n"
                f"Available actions: {opt_list}.\n\n"
                "- 'extract': only extract/inspect content from the file(s) without persisting to any database or knowledge base.\n"
                "- 'upsert': extract as needed and then upsert / save the content into the knowledge base.\n\n"
                f"User request: {user_prompt!r}\n\n"
                f"Return exactly one action key from: {opt_list}.\n"
                "Answer with the key only, no explanations."
            )
            resp = gem.ask(routing_prompt) or ""
            text = resp.strip().strip('"').strip("'").lower()
            for k in keys:
                if text == k.lower():
                    chosen_key = k
                    break
            if not chosen_key:
                # Fallback: substring match in case the model adds text.
                for k in keys:
                    if k.lower() in text:
                        chosen_key = k
                        break
        except Exception as e:
            print(f"[FileManager] _classify_and_call: Gem routing error: {e}")

        # 2) Heuristic fallback if LLM routing failed or is unavailable.
        if not chosen_key:
            lp = (user_prompt or "").lower()
            if any(w in lp for w in ["upsert", "save", "store", "remember", "kb", "knowledge base"]):
                for k in keys:
                    if k.lower() == "upsert":
                        chosen_key = k
                        break
            if not chosen_key:
                for k in keys:
                    if k.lower() == "extract":
                        chosen_key = k
                        break

        # 3) Final fallback: default key or first option.
        if not chosen_key:
            chosen_key = default_key if default_key in options else keys[0]

        print(f"[FileManager] _classify_and_call: chosen action={chosen_key}")
        func = options.get(chosen_key)
        if not callable(func):
            raise ValueError(f"[FileManager] _classify_and_call: chosen action '{chosen_key}' is not callable")
        return func()

    def route_file_action(
        self,
        user_id: str,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Route a file request based on the user text prompt.

        - 'extract' => extract-only (no upsert; uses testing=True).
        - 'upsert'  => full pipeline including upsert to the knowledge base.
        """
        user_prompt = (data.get("prompt") or data.get("msg") or "").strip()
        options: Dict[str, Callable[[], Any]] = {
            # Extract-only: skip BigQuery upserts via testing=True.
            "extract": lambda: self.process_and_upload_file_config(
                user_id=user_id,
                data=data,
                testing=True,
                mock_extraction=False,
            ),
            # Upsert: full pipeline with persistence.
            "upsert": lambda: self.process_and_upload_file_config(
                user_id=user_id,
                data=data,
                testing=False,
                mock_extraction=False,
            ),
        }
        return self._classify_and_call(user_prompt=user_prompt, options=options, default_key="upsert")

    def set_module(self, rows: List[Dict] or Dict, user_id: str):
        """
        Upsert module entry to BigQuery.
        Direct copy/adapt from ModuleWsManager.
        """
        print("[FileManager] set_module: starting")
        if isinstance(rows, dict):
            rows = [rows]

        for row in rows:
            row["user_id"] = user_id
            
            if row.get("parent"):
                row.pop("parent")

            if row.get("module_index"):
                row.pop("module_index")

        # Serializing specific fields is handled in QBrainTableManager.set_item but 
        # let's duplicate the safety check if needed or rely on qb.
        self.qb.set_item(self.MODULES_TABLE, rows)
        print(f"[FileManager] set_module: done -> {len(rows)} row(s)")

 

# Default instance for standalone use (no orchestrator context)
_default_bqcore = BQCore(dataset_id="QBRAIN")
_default_file_manager = FileManager(get_qbrain_table_manager(_default_bqcore))
file_manager = _default_file_manager  # backward compat

# -- VALIDATION HANDLERS --

def handle_set_file(data=None, auth=None):
    """Process file config and upsert module. Required: user_id (auth), data (file config). Optional: original_id (auth)."""
    from qbrain.core.managers_context import get_file_manager
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    file_data = data if isinstance(data, dict) else None
    original_id = get_val(data, auth, "original_id")
    if err := require_param(user_id, "user_id"):
        return err
    if err := require_param_truthy(file_data, "data"):
        return err
    fm = get_file_manager()
    if original_id:
        try:
            fm.qb.del_entry(original_id, "modules", user_id)
        except Exception:
            pass
    return fm.process_and_upload_file_config(user_id, file_data)


def handle_route_file(data=None, auth=None):
    """Route file action by user intent. Required: user_id (auth). Optional: data (prompt, files, etc.)."""
    from qbrain.core.managers_context import get_file_manager
    data, auth = data or {}, auth or {}
    user_id = get_val(data, auth, "user_id")
    file_data = data if isinstance(data, dict) else {}
    if err := require_param(user_id, "user_id"):
        return err
    return get_file_manager().route_file_action(user_id=user_id, data=file_data)

if __name__ == "__main__":
    _default_file_manager.process_and_upload_file_config(
        user_id="public",
        data={
            "id": "hi",
            "files": [
                open(r"C:\Users\bestb\PycharmProjects\BestBrain\test_paper.pdf", "rb")
            ]
        },
        testing=False
    )