from .method_lib import (
    handle_list_users_methods,
    handle_get_sessions_methods,
    handle_link_session_method,
    handle_rm_link_session_method,
    handle_del_method,
    handle_set_method,
    handle_get_method
)

RELAY_METHOD = [
    {
        "case": "LIST_USERS_METHODS",
        "desc": "List User's Methods",
        "func": handle_list_users_methods,
        "req_struct": {
            "auth": {
                "user_id": "str"
            }
        },
        "out_struct": {
            "type": "LIST_USERS_METHODS",
            "data": {
                "methods": "list"
            }
        }
    },
    {
        "case": "GET_USERS_METHODS",
        "desc": "Get User's Methods (alias)",
        "func": handle_list_users_methods,
        "req_struct": {
            "auth": {
                "user_id": "str"
            }
        },
        "out_struct": {
            "type": "LIST_USERS_METHODS",
            "data": {
                "methods": "list"
            }
        }
    },
    {
        "case": "GET_SESSIONS_METHODS",
        "desc": "Get Session Methods",
        "func": handle_get_sessions_methods,
        "req_struct": {
            "auth": {
                "user_id": "str",
                "session_id": "str"
            }
        },
        "out_struct": {
            "type": "GET_SESSIONS_METHODS",
            "data": {
                "methods": "list"
            }
        }
    },
    {
        "case": "LINK_SESSION_METHOD",
        "desc": "Link Session Method",
        "func": handle_link_session_method,
        "req_struct": {
            "auth": {
                "user_id": "str",
                "method_id": "str",
                "session_id": "str"
            }
        },
        "out_struct": {
            "type": "GET_SESSIONS_METHODS",
            "data": {
                "methods": "list"
            }
        }
    },
    {
        "case": "RM_LINK_SESSION_METHOD",
        "desc": "Remove Link Session Method",
        "func": handle_rm_link_session_method,
        "req_struct": {
            "auth": {
                "user_id": "str",
                "method_id": "str",
                "session_id": "str"
            }
        },
        "out_struct": {
            "type": "GET_SESSIONS_METHODS",
            "data": {
                "methods": "list"
            }
        }
    },
    {
        "case": "DEL_METHOD",
        "desc": "Delete Method",
        "func": handle_del_method,
        "req_struct": {
            "auth": {
                "user_id": "str",
                "method_id": "str"
            }
        },
        "out_struct": {
            "type": "LIST_USERS_METHODS",
            "data": {
                "methods": "list"
            }
        }
    },
    {
        "case": "SET_METHOD",
        "desc": "Set/Update Method with JAX Code Generation",
        "func": handle_set_method,
        "req_struct": {
            "data": {
                "id": "str",
                "equation": "str",
                "params": "list",
                "origins": "list",
                "description": "str",
                "code": "str",
                "jax_code": "str"
            },
            "auth": {
                "user_id": "str",
                "original_id": "str"
            }
        },
        "out_struct": {
            "type": "LIST_USERS_METHODS",
            "data": {
                "methods": "list"
            }
        }
    },
    {
        "case": "GET_METHOD",
        "desc": "Get Method by ID",
        "func": handle_get_method,
        "req_struct": {
            "auth": {
                "method_id": "str"
            }
        },
        "out_struct": {
            "type": "GET_METHOD",
            "data": "dict"
        }
    }
]
