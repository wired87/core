import json
import base64
import logging
import pprint
import networkx as nx
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import dotenv
import filetype

from core.module_manager.mcreator import ModuleCreator
from core.file_manager.extraction_prompts import EXTRACT_EQUATIONS_PROMPT, CONV_EQ_CODE_TO_JAX_PROMPT
from qf_utils.qf_utils import QFUtils

dotenv.load_dotenv()

from core.app_utils import GCP_ID
from auth.load_sa_creds import load_service_account_credentials

class RawModuleExtractor:

    def __init__(self):
        try:
            vertexai.init(
                project=GCP_ID, 
                location="us-central1", 
                credentials=load_service_account_credentials()
            ) 
            self.model = GenerativeModel("gemini-2.5-pro")
        except Exception as e:
            logging.error(f"Failed to init Vertex AI: {e}")
            self.model = None

    def _prepare_content_parts(self, files: list[str]):
        """
        Processes a list of stringified byte strings, detects their type,
        and returns a list of Vertex AI Part objects.
        """
        print("_prepare_content_parts...")

        classified_files = self._classify_files(files)
        print("classified_files", classified_files)

        parts = []

        for ftype, file_list in classified_files.items():
            print(f"ftype, file_list", ftype, file_list)

            mime_type = self._get_mime_type(ftype)

            for f_bytes in file_list:
                try:
                    parts.append(
                        Part.from_data(f_bytes, mime_type=mime_type)
                    )
                except Exception as e:
                    logging.error(f"Error creating Part for {ftype}: {e}")
        print(f"Content parts prepared. Classified types: {list(classified_files.keys())}")
        return parts, classified_files


    def _classify_files(self, files: list[str]) -> dict:
        """
        Classifies files based on their content (bytes).
        Input: List of stringified byte strings (or base64 strings).
        Output: Dictionary { 'pdf': [bytes, ...], 'image': [bytes, ...] }
        """
        classified = {}
        for f_str in files:
            try:
                # Handle base64 prefix if present
                if isinstance(f_str, str):
                    if "base64," in f_str:
                         f_str = f_str.split("base64,")[1]
                    f_bytes = base64.b64decode(f_str)
                else:
                    f_bytes = f_str

                # Detect type
                kind = filetype.guess(f_bytes)
                if kind:
                    ftype = kind.extension
                else:
                    # Fallback or default
                    ftype = "unknown"
                
                if ftype not in classified:
                    classified[ftype] = []
                classified[ftype].append(f_bytes)
                
            except Exception as e:
                 logging.error(f"Error classifying file: {e}")
        
        return classified

    def _get_mime_type(self, extension: str) -> str:
        """Maps extensions to MIME types."""
        mimes = {
            "pdf": "application/pdf",
            "png": "image/png",
            "jpg": "image/jpeg",

            "jpeg": "image/jpeg",
            "txt": "text/plain",
            "csv": "text/csv"
        }
        return mimes.get(extension, "application/octet-stream")


    def extract_params_and_data_types(self, parts):
        if not self.model or not parts:
            return {}
            
        # Add prompt to parts
        prompt = "Extract all parameters used in equations from the following documents. Return a dictionary mapping parameter names to BigQuery data types (e.g. FLOAT64, STRING). Output valid JSON only."
        request_parts = parts + [prompt]
        try:
            response = self.model.generate_content(request_parts)
            text = response.text
            text = text.replace("```json", "").replace("```", "").strip()
            params = json.loads(text)
            print(f"Extracted params: {params}")
            return params
        except Exception as e:
            logging.error(f"Gemini extract params error: {e}")
            return {}

    def extract_equations(self, parts):
        print("_extract_equations...")
        request_parts = parts +[EXTRACT_EQUATIONS_PROMPT]
        try:
            print("_extract_equations... request_parts")
            response = self.model.generate_content(request_parts)
            print("_extract_equations... response", response)
            text = response.text.strip()
            text = text.replace("```python", "").replace("```", "").strip()
            print(f"Extracted equations code length: {len(text)}:", text)
            return text
        except Exception as e:
             logging.error(f"Gemini extract equations error: {e}")
             return ""

    def jax_predator(self, code):
        prompt = CONV_EQ_CODE_TO_JAX_PROMPT
        prompt += f"""\n\nPYTHON CODE: {code}"""

        try:
             response = self.model.generate_content(prompt)
             text = response.text.strip()
             text = text.replace("```python", "").replace("```", "").strip()
             print(f"Generated JAX code length: {len(text)}:", text)
             return text
        except Exception as e:
             logging.error(f"Gemini jax predator error: {e}")
             return code 

    def process(self, mid:str, files: list[str]):
        """
        Main workflow: process files -> extract params -> extract equations -> optimize.
        Returns extracted data structure.
        """
        G = nx.Graph()
        self.mcreator = ModuleCreator(
            G=G,
            qfu=QFUtils(G=G),
        )

        # 1. Prepare parts once
        parts, classified_files = self._prepare_content_parts(files)
        print("1. Files prepared.")

        # 3. Extract Equations
        print("2. Extracting equations...")
        code = self.extract_equations(parts)
        
        # MODULE STUFF
        print("3. Processing module graph...")
        self.mcreator.create_modulator(
            mid, code
        )

        params_edges = self.mcreator.qfu.g.get_neighbor_list(
            node=mid,
            target_type="PARAM",
        )

        params = {
            p["trgt"]: p["attrs"].get("type", "Any")
            for p in params_edges.values()
        }
        
        # EXTRACT METHODS
        method_edges = self.mcreator.qfu.g.get_neighbor_list(
            node=mid,
            target_type="METHOD",
        )
        methods = [m["attrs"] for m in method_edges.values()]

        # EXTRACT FIELDS (CLASS_VAR)
        field_edges = self.mcreator.qfu.g.get_neighbor_list(
            node=mid,
            target_type="CLASS_VAR",
        )
        fields = [f["attrs"] for f in field_edges.values()]

        print(f"Module params identified: {params}")
        print(f"Module methods identified: {len(methods)}")
        print(f"Module fields identified: {len(fields)}")

        # 4. Optimize
        print("4. Optimizing with JAX...")
        jax_code = self.jax_predator(code)
        
        # 5. Return Data
        data= {
            "params": params,
            "methods": methods,
            "fields": fields,
            "code": code,
            "jax_code": jax_code,
        }
        print("data extracted:")
        pprint.pp(data)
        return data
