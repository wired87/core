

class DataMessageManager:

    """
    Receives DataUpdates for a single sfn
    """

    def __init__(self, parent_ref):
        self.parent = parent_ref


    def get_data_update(self, payload):
        attrs = payload["data"]
        self.parent.main.remote()
