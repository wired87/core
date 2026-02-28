from qbrain.core.qbrain_manager import get_qbrain_table_manager

if __name__ == "__main__":
    qbrain_manager = get_qbrain_table_manager()
    qbrain_manager.reset_tables(
        [
            "methods",
            "fields",
            "modules",
            #"users",
            "params",
            #"injections",
        ]
    )


