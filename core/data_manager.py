import ray

from app_utils import USER_ID, ENV_ID
from fb_core.fb_to_table import TimeSeriesFBConverter
from fb_core.real_time_database import FBRTDBMgr


class DataManager:

    def __init__(self):
        self.data_converter = TimeSeriesFBConverter()
        self.db_manager = FBRTDBMgr()

        # VISUALIZER -> just energy distribution
        self.visualizer = ray.get_actor(
            name="UTILS_WORKER"
        ).update_ndata.remote()

        # TRAINER -> all t-steps
        self.trainer = ray.get_actor(
            name="TRAINER"
        ).update_ndata.remote()

        self.fb_endp = f"users/{USER_ID}/env/{ENV_ID}/datastore/"

    def handle_data(self, path):

        data = self.db_manager.get_data(
            path=path
        )

        for nid, entries in data.items():
            # convert and save ts data csv
            file_name = f"{nid}.csv"
            self.data_converter.convert_to_tables(
                entries,
                file_name,
            )

        # gnn






