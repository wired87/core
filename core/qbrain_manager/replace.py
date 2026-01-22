from core.qbrain_manager import QBrainTableManager

if __name__ == "__main__":
    qbrain_manager = QBrainTableManager()
    qbrain_manager.reset_tables(["fields", "methods", "params"])
