# Prompt: Module Manager - Center Field Identification

**Source**: `qbrain/core/module_manager/utils/prompts.py` – `CENTER_FIELD_PROMPT`

Identify the MAIN FIELD discussed in the provided research text.

## YOU MUST

1. Detect all fields: scalar, fermion, gauge, tensor, synthetic fields.
2. Determine which field is the central object of the analysis.
   **Criteria**:
   - Appears in key equations frequently.
   - Appears in mass, interaction, or kinetic terms.
   - Mentioned as "we study", "the model contains", "the field", etc.

3. Compare the inferred field name to the list 'allowed_fields' (provided externally).

## BEHAVIOR

- If a match exists → return that canonical field name.
- If no match → return the inferred field name and treat it as synthetic.

## OUTPUT FORMAT (JSON)

```json
{
  "center_field": "<name>"
}
```
