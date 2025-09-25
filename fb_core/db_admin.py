import os
import pprint

from app_utils import USER_ID, ENV_ID
from fb_core.real_time_database import FirebaseRTDBManager


class DBAdmin:

    def __init__(self, user_id=USER_ID, env_id=ENV_ID):
        self.user_id = user_id
        self.env_id = env_id
        self.database = f"users/{self.user_id}/env/{self.env_id}"
        self.metadata_path = "metadata"
        self.states_path = "global_states"

        self.db_manager = FirebaseRTDBManager()
        self.db_manager.set_root_ref(self.database)

    def change_state(self, state=None):
        """Changes state of ALL metadata entries"""
        upsert_data = {}
        data = self.db_manager.get_data(path=self.metadata_path)
        ready = None
        for mid, data in data["metadata"].items():
            if state is None:
                current_state = data["status"]["state"]
                if current_state == "active":
                    new_state = "inactive"
                    if ready is None:
                        ready=False
                else:
                    new_state = "active"
                    if ready is None:
                        ready = True
            else:
                new_state = state
                ready = False

            upsert_data[f"{mid}/status/state/"] = new_state

        #pprint.pp(upsert_data)

        self.db_manager.update_data(
            path=self.metadata_path,
            data=upsert_data
        )


    def delete_process(self):
        try:
            success:bool = self.db_manager.delete_data(path="/")
            if success is not True:
                raise ValueError()
            print("Data deleted")
        except Exception as e:
            # get env keys
            print(f"Delete Error: {e}")
            data = self.db_manager.get_data(ref_root="users/rajtigesomnlhfyqzbvx/", path="env/")
            #pprint.pp(data)
            for root, env in data.items():
                print(f"ENV: {env.keys()}")
                for env_root, struct in env.items():
                    for env_id, stuff in struct.items():
                        print(f"Delete data from:{env_id}")
                        if env_id is not None:
                            self.db_manager.delete_data(path=f"users/rajtigesomnlhfyqzbvx/env/{env_id}/")
                            print(f"Deleted data from {env_id}")


if __name__ == "__main__":
    admin = DBAdmin(env_id="env_rajtigesomnlhfyqzbvx_qjodlmdexctwpfvaeqan")
    admin.delete_process()
    #pprint.pp(admin.db_manager.get_data(path="cfg"))
    #admin.change_state(state="inactive")
