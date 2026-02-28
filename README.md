# BestBrain

## Workflow Mastermap

- Full visual workflow + action catalog: `docs/PROJECT_WORKFLOW_MASTERMAP.md`

## Current Core TODOs (from existing intent)

- [ ] `get_data`: integrate BigQuery -> Sheets live data view (`table=ntype`, `col=px`, `row=ts state`).
- [ ] Improve method extraction (bracket parsing and dedupe) and reliably inject method defs into `Guard.method_layer`.

## Near-Term Engine Priorities

- [ ] Implement relay case consumption hardening and payload contract validation.
- [ ] Add Guard answer caching and cross-module parameter/field consistency checks.
- [ ] Add observability (latency, error rates, case-level tracing) for Relay/Orchestrator/Guard.