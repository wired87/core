import ray

from qbrain.core.app_utils import USER_ID, ENV_ID
from fb_core.fb_to_table import TimeSeriesFBConverter
from fb_core.real_time_database import FBRTDBMgr

_DATA_DEBUG = "[DataManager]"


class DataManager:

    def __init__(self):
        try:
            print(f"{_DATA_DEBUG} __init__: initializing")
            self.data_converter = TimeSeriesFBConverter()
            self.db_manager = FBRTDBMgr()
            self.visualizer = ray.get_actor(name="UTILS_WORKER").update_ndata.remote()
            self.trainer = ray.get_actor(name="TRAINER").update_ndata.remote()
            self.fb_endp = f"users/{USER_ID}/env/{ENV_ID}/datastore/"
            print(f"{_DATA_DEBUG} __init__: done")
        except Exception as e:
            print(f"{_DATA_DEBUG} __init__: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def handle_data(self, path):
        try:
            print(f"{_DATA_DEBUG} handle_data: path={path}")
            data = self.db_manager.get_data(path=path)
            for nid, entries in data.items():
                file_name = f"{nid}.csv"
                self.data_converter.convert_to_tables(entries, file_name)
            print(f"{_DATA_DEBUG} handle_data: done")
        except Exception as e:
            print(f"{_DATA_DEBUG} handle_data: error: {e}")
            import traceback
            traceback.print_exc()
            raise
