# Module Converter Specification

## MODULE CONVERTER
**Status:** # todo frontend

### Event Handler ()
**Case:** `receive type="CONVERT_MODULE", auth={module_id:str}, data={files={pdf:list[pdf-files]}`

---

### Class: RawModuleExtractor

This class handles the extraction, transformation, and storage of mathematical modules from PDF documentation.

#### 1. extract_params_and_data_types
* **Action:** Send Request to Gemini 3.5.
* **Goal:** Extract all parameters used in equations within the provided files.
* **Mapping:** Link parameters to BigQuery-compliant data types.
* **Output:** `dict[param_key:str, bigquery_type:str]`

#### 2. extract_equations
* **Input:** `param_struct` (Output of `extract_params_and_data_types`).
* **Action:** Send Request to Gemini 3.5.
* **Goal:** Extract and convert equations into executable Python functions (compiled string format).
* **Rules:**
    * Must be serializable.
    * Strictly use only specified parameters.
    * Nested functions (avoid if possible) must start with `_`.
* **Output:** `{code: "executable code string"}`

#### 3. jax_predator
* **Input:** `code` (Output of `extract_equations`).
* **Action:** Optimize code for JAX GPU-based processing.
* **Rules:**
    * Maintain persistency.
    * Optimized for production environment.
* **Output:** `{code: "jax optimized executable codebase"}`

#### 4. upsert_module
* **Action:** Upsert the module data to the `modules-table` within the `QBRAIN` dataset in BigQuery.

---

### websocket Response ithin raly_station receive
**Return:** `type=CONVERT_MODULE`,  
`data={params: extract_params_and_data_types return, code: jax_predator return}`