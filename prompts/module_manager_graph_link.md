# Prompt: Module Manager - Graph Link Extraction

**Source**: `qbrain/core/module_manager/utils/prompts.py` – `GRAPH_LINK_PROMPT`

Extract all fields that INTERACT with each other in the given research text.

## DEFINITION OF INTERACTION

- Any term where multiple fields multiply.
- Derivative couplings.
- Gauge interactions.
- Yukawa terms.
- Scalar potential terms.
- Field-strength dependencies.
- Mixing or mass-mixing.

## REQUIREMENTS

- Return only FIELD NAMES (no parameters, no constants).
- Do not include derivatives as separate fields.
- Do not include indices.
- Remove duplicates.
- Include synthetic fields if they appear.

## OUTPUT FORMAT (JSON)

```json
{
  "graph_links": ["field1", "field2", ...]
}
```
