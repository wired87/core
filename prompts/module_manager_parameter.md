# Prompt: Module Manager - Parameter Extraction

**Source**: `qbrain/core/module_manager/utils/prompts.py` – `PARAMETER_PROMPT`

Extract ALL physical parameters from the provided research text.

## DEFINITION OF PARAMETERS

- Couplings (g, y, λ, κ, etc.)
- Mass terms (m, m_phi, m_psi, etc.)
- Charges, constants, mixing angles
- Cutoff scales, vacuum expectation values (vev)
- Anything declared as: "parameter", "constant", "coupling", "mass", "coefficient".

## RULES

- NO fields (psi, phi, A_mu, etc.) unless explicitly defined as constants.
- NO indices.
- Return each parameter with a short type description (1–4 words).
- Keep original symbol exactly.

## OUTPUT FORMAT (JSON)

```json
{
  "parameters": [
    { "name": "<symbol>", "type": "<short type description>" }
  ]
}
```
