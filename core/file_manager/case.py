from .file_lib import handle_set_file

RELAY_FILE = [
    {
        "case": "SET_FILE",
        "desc": "Set File (Module from File)",
        "func": handle_set_file,
        "req_struct": {
            "data": {"id": "str", "files": "list"},
            "auth": {"user_id": "str"}
        },
        "out_struct": {
            "type": "LIST_USERS_MODULES",
            "data": {"modules": "list"}
        }
    }
]
