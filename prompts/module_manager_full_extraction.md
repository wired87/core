# Prompt: Module Manager - Full Physics Extraction

**Source**: `qbrain/core/module_manager/utils/prompts.py` – `FULL_EXTRACTION_PROMPT`

Perform a structured physics-paper extraction.

## TASKS

### 1) Equations

- Extract all equations.
- Convert to valid Python code.
- Format: `"equations": [ {"original": "...", "python": "..."} ]`

### 2) Parameters

- Extract all physical parameters (masses, couplings, constants).
- Format: `"parameters": [ {"name": "...", "type": "..."} ]`

### 3) Center Field

- Identify the main field.
- Compare with allowed_fields (provided externally).
- If match, return matched name; else synthetic field.
- Format: `"center_field": "<name>"`

### 4) Graph Links

- Extract all interacting fields.
- Return unique field names only.
- Format: `"graph_links": ["...", "..."]`

Return final structured result as JSON.
