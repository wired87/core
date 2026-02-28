# WebSocket Route Alignment: Backend ↔ qdash Frontend

This document describes the alignment between relay case structs (and their handlers) and the qdash frontend WebSocket message handlers.

## Summary of Changes

### Environment (ENV)
| Case | Before | After | Frontend Handler |
|------|--------|-------|------------------|
| SET_ENV | `GET_USERS_ENVS` | `SET_ENV_SUCCESS` with `env_id`, `config`, `data.envs` | ENV_SET / SET_ENV_SUCCESS |
| DEL_ENV | `GET_USERS_ENVS` | `DEL_ENV` with `env_id`, `data.envs` | ENV_DELETED / DEL_ENV |

### Session (SESSION)
| Case | Before | After | Frontend Handler |
|------|--------|-------|------------------|
| LINK_ENV_SESSION | `LIST_SESSIONS_ENVS` | `LINK_ENV_SESSION` with `auth`, `data.sessions` | LINK_ENV_SESSION |
| RM_LINK_ENV_SESSION | `LIST_SESSIONS_ENVS` | `RM_LINK_ENV_SESSION` with `auth` (session_id, env_id), `data` | RM_LINK_ENV_SESSION |

### Module (MODULE)
| Case | Before | After | Frontend Handler |
|------|--------|-------|------------------|
| RM_LINK_ENV_MODULE | `LINK_ENV_MODULE` | `RM_LINK_ENV_MODULE` with `auth` (session_id, env_id, module_id), `data` | RM_LINK_ENV_MODULE |

### Field (FIELD)
| Case | Before | After | Frontend Handler |
|------|--------|-------|------------------|
| LINK_MODULE_FIELD | `ENABLE_SM` (when session) / `GET_MODULES_FIELDS` | `LINK_MODULE_FIELD` with `auth`, `data.sessions` + `data.fields` | LINK_MODULE_FIELD |
| RM_LINK_MODULE_FIELD | `GET_MODULES_FIELDS` | `RM_LINK_MODULE_FIELD` with `auth`, `data` | RM_LINK_MODULE_FIELD |

### Injection (INJECTION)
| Case | Before | After | Frontend Handler |
|------|--------|-------|------------------|
| SET_INJ | `GET_INJ_USER` | `SET_INJ` with `status.state`, `data.id`, `data.injections` | SET_INJ |
| DEL_INJ | `GET_INJ_USER` | `DEL_INJ` with `status.state`, `data.id`, `data.injections` | DEL_INJ |

## Frontend Message Types (qdash websocket.js)

The frontend handles these message types. Backend responses are aligned to match:

- **ENV_SET / SET_ENV_SUCCESS** – env saved; expects `env_id`, `config`, optionally `data.envs`
- **DEL_ENV / ENV_DELETED** – env deleted; expects `env_id`, optionally `data.envs`
- **GET_USERS_ENVS / LIST_ENVS / GET_ENV / ...** – env list; expects `data.envs` or `data.environments`
- **LINK_ENV_SESSION** – env linked to session; expects `data.sessions`
- **RM_LINK_ENV_SESSION** – env unlinked; expects `auth.session_id`, `auth.env_id`, `data.sessions`
- **LINK_ENV_MODULE** – module linked; expects `data.sessions`
- **RM_LINK_ENV_MODULE** – module unlinked; expects `auth.session_id`, `auth.env_id`, `auth.module_id`
- **LINK_MODULE_FIELD** – field linked; expects `data.sessions`, `data.fields`
- **RM_LINK_MODULE_FIELD** – field unlinked; expects `auth` (module_id, field_id, optionally session_id, env_id)
- **SET_INJ** – injection saved; expects `status.state`, `data.id`
- **DEL_INJ** – injection deleted; expects `status.state`, `data.id`

## Files Modified

- `core/env_manager/env_lib.py` – handle_set_env, handle_del_env
- `core/env_manager/case.py` – out_struct for SET_ENV, DEL_ENV
- `core/session_manager/session.py` – handle_link_env_session, handle_rm_link_env_session
- `core/session_manager/case.py` – out_struct for RM_LINK_ENV_SESSION
- `core/env_manager/env_lib.py` – handle_rm_link_env_module
- `core/module_manager/ws_modules_manager/case.py` – out_struct for RM_LINK_ENV_MODULE
- `core/fields_manager/fields_lib.py` – handle_link_module_field, handle_rm_link_module_field
- `core/injection_manager/injection.py` – handle_set_inj, handle_del_inj
- `qdash/src/qdash/websocket.js` – handlers for SET_ENV_SUCCESS, DEL_ENV, SET_INJ, DEL_INJ (data.envs, data.injections)
