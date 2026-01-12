import json
import os
import time

import ray

from _ray_core.base._ray_utils import RayUtils
from core.app_utils import LOGGING_DIR, FB_DB_ROOT, GLOBAC_STORE
from _ray_core.base.base import BaseActor
from utils.last_folder_from_dir import get_last_modified_folder
from utils.logger import get_log_id

@ray.remote
class LoggerWorker(
    BaseActor
):
    #todo remove first lines form logs
    """
    One logger per pixel
    """

    def __init__(
            self,
            host={} # intial, hsot need to be received aftr creating all externs. for intern px, ost wih required can be passed here
    ):
        self.host = host or {}
        BaseActor.__init__(self)
        RayUtils.__init__(self)
        self.logging_urls = None
        self.json_logs = os.environ.get("RAY_LOGGING_CONFIG_ENCODING", "Hi") == "JSON"

        self.ray_root = r"C:\Users\bestb\Desktop\qfs\tmp\ray" if os.name == "nt" else "/tmp/ray/"
        self.logging_root = get_last_modified_folder(self.ray_root)

        """self.latest_session_path = "
        if os.path.isfile(self.latest_session_path):
            os.remove(self.latest_session_path)
        """
        self.dir = LOGGING_DIR
        self.database = FB_DB_ROOT
        self.retry = 0

        self.struct = {}
        # Finish init
        print("LOGGER initialized")

    async def handle_initialized(self, host):  # receive_utils
        print(f"LOGGER rcv host", host)
        if self.host is None:
            self.host = {}
        self.host.update(host)

    def ping(self):
        return True

    def main(self, worker_ids):
        try:
            print("Start Logger.main")
            self._set_worker_info(worker_ids)
            self.set_run(True)
            self.stream_loggs()
        except Exception as e:
            print(f"Exception while stsrting main: {e}")


    def _set_worker_info(self, worker_ids):
        #print("set worker info")
        for w in worker_ids:
            self.struct[w] = {
                    "lines": 0,
                    "lines_to_send": {},
                }
        print("finished set worker info")


    def get_actor_info(self, ref):
        """
        :return:
        - job_id
        - actor_id
        - task_id
        - ref
        """
        struct = {}
        struct["ref"] = ref
        struct.update(
            ray.get(
                ref.get_actor_info.remote()
            )
        )
        return struct


    def extract_pid(self, file_name):
        return file_name.split("-")[-1].split(".")[0]

    def get_ref(self, acor_id):
        try:
            ref = ray.get_actor(acor_id)
            # print("ref in logger extracted:", ref)
            return ref
        except Exception as e:
            print(f"Failed load actor form id {acor_id}: {e}")
            return None


    def set_run(self, run:bool):
        #print(f"set run: {run}")
        self.run = run

    def stream_loggs(self):
        print(f"Start logging from {self.logging_root}")

        if self.logging_urls is None:
            self.logging_urls: dict = self.get_logging_urls()

        while self.run is True:
            try:
                for nid, path_struct in self.logging_urls.items():
                    if nid in self.struct and ("LOGGER" not in nid or nid.split("__")[0] != "LOGGER"):
                        # extract err
                        for k, path in path_struct.items():
                            if k in ["out", "err"]:
                                content, file_path = self.validade_path(
                                    path,
                                    path_struct["fallback"],
                                    key=k
                                )
                                if content is not None:
                                    self.read_upsert_logs(
                                        content,
                                        nid,
                                        file_path,
                                        file_type=k
                                    )

            except Exception as e:
                print(f"Excepion while running logger: {e}")
                if self.retry <= 5:
                    self.retry += 1
                    time.sleep(1)

    def get_logging_urls(self):
        #print("Get logging urls")
        try:
            return ray.get(GLOBAC_STORE["UTILS_WORKER"].get_logging_urls.remote())
        except Exception as e:
            print(f"Err get_logging_urls {e}")

    def read_upsert_logs(self, f, nid, file_path, file_type:str = "out"):
        local_test = os.name == "nt"
        try:
            lines = f.readlines()
            file_len = len(lines)
            if file_len >= 1:
                if local_test is True:
                    self.extract_logs_keep_local(nid, file_len, lines, file_type)
                else:
                    self.extract_logs_delete_local(
                        nid, file_len, lines, file_type, file_path
                    )
        except Exception as e:
            #print(f"Error extracting log lines for {nid}: {e}")
            self.upsert(
                {"err": str(e)},
                nid,
                file_type
            )




    def extract_logs_delete_local(self, nid, file_len, lines, file_type, file_path):
        for i in range(file_len):
            fline = lines[i]
            if "job_id:" not in fline or "actor_name:" not in fline:
                self.upsert(
                    self.conv_log_content(fline, nid),
                    nid,
                    file_type
                )

            # delete existing lines
            with open(file_path, 'w') as f:
                pass


    def extract_logs_keep_local(self, nid, file_len, lines, file_type):
        """
        Extracts all logs and upserts them but keeps local
        """
        last_lines = self.struct[nid]["lines"]
        if file_len > last_lines:
            for i in range(last_lines, file_len):
                fline = lines[i]
                # exclude init lines
                if "job_id:" not in fline or "actor_name:" not in fline:
                    self.upsert(
                        self.conv_log_content(fline, nid),
                        nid,
                        file_type
                    )
            # update
            self.struct[nid]["lines"] += file_len - last_lines
    def conv_log_content(self, line, nid):
        if self.json_logs is False:
            fb_dest = f"{get_log_id(nid)}"
            return {
                fb_dest: line
            }
        else:
            log_content = json.loads(line) # -> jsonl format
            fb_dest = log_content.get("asctime").replace(":", "_").replace(" ", "_").replace("-", "_").replace(",", "_")
            return {
                fb_dest: log_content.pop("asctime")
            }

    def upsert(self, logs, nid, file_type):
        if "LOGGER" in nid or nid.split("__")[0] == "LOGGER":
            return
        try:
            """
            print(f"logs: {logs}")
            print(f"nid: {nid}")
            print(f"file_type: {file_type}")
            """
            self.host["DB_WORKER"].iter_upsert.remote(
                attrs=logs,
                path=f"{self.database}/logs/{nid}/{file_type}/",
            )
        except Exception as e:
            print(f"Logger.upsert failed: {e}")


    def validade_path(self, path, fallback, key):
        """Extract log file content """
        content = None
        path = os.path.join(self.logging_root, path)
        try:
            content = open(path, "r")
        except Exception as e:
            print(f"Error extracting: {e}")
            try:
                path = os.path.join(self.logging_root, f"{fallback}.{key}")
                content = open(path, "r")
            except Exception as e:
                print(f"Error 2 extracting: {e}")
        return content, path
