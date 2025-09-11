import os
import re

import firebase_admin
from firebase_admin import db
import logging # Good practice for backend applications

from dotenv import load_dotenv
from firebase_admin.db import Reference

from utils.auth import AuthManager
from qf_core_base.qf_utils.all_subs import ALL_SUBS

load_dotenv()
DB_URL = os.environ.get("FIREBASE_RTDB")

# DS PATH             fb_dest=f"users/{self.user_id}/datastore/{self.envc_id}/",
# G STATE PATH             fb_dest=f"users/{self.user_id}/env/{self.envc_id}/",
# GLOBAL STATE PATH f"{self.database}/global_states/"


# todo alle ds werden in gleichen apth geuppt (keine extra sessions) (vorerst)


class FirebaseRTDBManager(AuthManager):
    """
    A class to manage interactions with Firebase Realtime Database
    from a Python backend using the Admin SDK.

    Schemas:
    users/user_id/env/env_id/objects/qf/qfn_ids
    """

    def __init__(self):
        """
        Initializes the Firebase Admin SDK and gets a database reference.

        Args:
            service_account_key_path: Path to your Firebase Service Account JSON key file.
            database_url: The URL of your Firebase Realtime Database (e.g., 'https://YOUR-PROJECT-ID-default-rtdb.europe-west1.firebaseio.com/').
        """

        AuthManager.__init__(self, auth=["fb"])
        self.db_url = DB_URL

        print("Firebase url:", self.db_url)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(self.creds["fb"], {
                'databaseURL': DB_URL
            })
            logging.info("Firebase Admin SDK initialized successfully.")


        self.invalid_keys_detected = []

    def set_root_ref(self, base_path):
        self.root_ref = db.reference(base_path)

    def _get_ref(self, path: str):
        """Helper to get a database reference for a specific path."""
        if not path or path == '/':
            return self.root_ref
        # Ensure path starts without a leading slash if concatenating
        if path.startswith('/'):
             path = path[1:]
        return self.root_ref.child(path)





    def upsert_data(
            self,
            path: str,
            data: dict,
            list_entry=False,
    ):
        """
        Inserts or updates data at the specified path.
        Equivalent to set(). If the path exists, it's overwritten.
        If parent paths don't exist, they are created.

        Args:
            path: The path in the database (e.g., '/users/alice').
            data: The data (as a dictionary) to write.
        Returns:
            True on success, False on failure.
        """

        try:

            #print("Upsert data:")
            #pprint.pp(data)

            if list_entry is True:
                db.reference(path).push(data)
            else:
                db.reference(path).update(data)

            print(f"Successfully upserted data")
            return True
        except Exception as e:
            print(f"Failed to upsert data at path {path}: {e}")
            return False

    def push_list_item(self, path, item):
        ref = db.reference(path)
        ref.push(item)
        print(f"Neues Element erfolgreich hinzugefügt unter Schlüssel: {path}")

    def get_latest_entries(self, path):
        """
        Ruft die 30 neuesten Einträge aus einem RTDB-Verzeichnis ab.
        """
        ref = db.reference(path)

        # Query erstellen: Sortieren nach Key und auf 30 begrenzen
        query = ref.order_by_key().limit_to_last(30)

        # Ausführen der Query
        snapshot = query.get()

        return snapshot

    def update_data(self, path: str, data: dict):
        try:
            ref = self._get_ref(path)
            ref.update(data)
            logging.info(f"Successfully updated data at path: {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to update data at path {path}: {e}")
            return False

    def delete_data(self, path: str):
        """
        Deletes data at the specified path relative to root_ref.

        Args:
            path: The path relative to the root_ref (e.g., 'users/alice').
        Returns:
            True on success, False on failure.
        """
        try:
            ref = db.reference(path)
            ref.delete()
            print(f"Successfully deleted data at path: {path}")
            return True
        except Exception as e:
            print(f"Failed to delete data at path {path}: {e}")
            return False

    def get_data(self, path: str or list):
        """
        Retrieves data from the specified path.

        Args:
            path: The path in the database (e.g., '/users/alice').
        Returns:
            The data at the path (as a dictionary or other Python type)
            or None if the path does not exist or on failure.
        """
        if isinstance(path, str):
            path = [path]
        try:
            sub_data = {}
            for p in path:
                print("Request data from", p)
                ref:Reference = db.reference(p)

                print("ref", ref._pathurl)
                data = ref.get()
                if data is not None:
                    print(f"Successfully retrieved data from path: {ref._pathurl}:")
                else:
                     print(f"No data found at path: {path}")

                if isinstance(data, tuple):
                    print("RECEIVED DATA AS TUPLE ")
                    data = data[0]
                sub_data[p] = data
            return sub_data
        except Exception as e:
            print(f"Failed to retrieve data from path {path}: {e}")
            return None

    def upsert_firebase(
            self,
            G,
            fb_dest=None,
            datastore=False
    ):
        print(f"Upsert G: {G} to FireBase: {fb_dest}")
        if datastore is False:
            updates = {}
            for nid, attrs in [(nid, attrs) for nid, attrs in G.nodes(data=True) if attrs.get("type") not in ["USERS"]]:
                #new_item = self._check_keys(attrs)

                path = f"{attrs.get('type')}/{nid}/"

                #pprint.pp(update_item)
                updates[path] = attrs

            for src, trgt in G.edges():
                if src is not None and trgt is not None:
                    edge_attrs = G[src][trgt]
                    eid = edge_attrs.get("id")
                    if eid:
                        path = f"edges/{eid}/"
                        updates[path] = edge_attrs
                    else:
                        raise ValueError(f"Edge {src} -> {trgt} has no id field ({edge_attrs})")

            # print("updates", updates)
            print("self.invalid_keys_detected", self.invalid_keys_detected)

        else:
            updates = {}
            """
            Upsert all start ds entries as list with one entry            
            """
            for time_nid, attrs in [(time_nid, attrs) for time_nid, attrs in G.nodes(data=True) if attrs.get("type") not in ["USERS"]]:
                nid = attrs.get("type")
                type = attrs.get("base_type")
                graph_type = attrs.get("graph_item")
                if graph_type == "node" and type is not None:
                    updates[f"{type}/{nid}/"] = attrs

        self.upsert_data(fb_dest, data=updates, list_entry=datastore)


    def _check_keys1(self, attrs, exclude:list = None):
        new_item = {}

        for k, v in attrs.items():
            # Iterate structs
            if isinstance(v, dict):
                attrs[k]=self._check_keys(attrs[k])
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        self._check_keys(item)

            # Exclude keys specified
            if exclude is not None and isinstance(exclude, list):
                if k in exclude:
                    continue
            for inv_char in ["/", "\\", ".", ",", ":", ";", "?", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "-",
                             "+", "=", "{", "}", "[", "]", "|", "<", ">", "`", "~"]:
                if inv_char in k:
                    print("INVALID KEY:", k)

                    if inv_char == "/":
                        k = k.replace(inv_char, "-")
                    else:
                        k = k.replace(inv_char, "_")

                    self.invalid_keys_detected.append(k)
                elif len(k) == 0:
                    continue
                new_item[k] = v
        return new_item

    def sanitize_firebase_key(self, key: str) -> str:
        """
        Sanitizes a string to be a valid Firebase database key.

        Firebase keys cannot contain '.', '$', '#', '[', ']', or '/'.
        This function replaces all invalid characters with an underscore '_'.

        Args:
            key: The original key string.

        Returns:
            The sanitized key string.
        """
        # Define the pattern for invalid characters
        invalid_chars_pattern = re.compile(r'[.#$/\[\]]')

        # Replace all invalid characters with an underscore
        sanitized_key = invalid_chars_pattern.sub('_', key)

        return sanitized_key

    def _check_keys(self, data: dict) -> dict:
        """
        Recursively sanitizes all keys in a dictionary and its nested dictionaries.

        Args:
            data: The dictionary to sanitize.

        Returns:
            A new dictionary with sanitized keys.
        """
        sanitized_data = {}
        for key, value in data.items():
            new_key = self.sanitize_firebase_key(key)
            if isinstance(value, dict):
                # Recursively sanitize nested dictionaries
                sanitized_data[new_key] = self._check_keys(value)
            else:
                sanitized_data[new_key] = value

        return sanitized_data


    def get_listener_endpoints(self, nodes:list[str], metadata=False):
        """
        :return: end for all given nodes to listen to state changes
        """
        return [
            f"{self.db_url}/{nid}" + "/metadata/status/" if metadata is True else None
            for nid in nodes
        ]

    def _get_db_paths_from_G(self, G, db_base, edges=True):
        # get paths for each node to lsiten to
        listener_paths = {
            "nodes": [],
            "edges": [],
            "meta": [],
        }

        for nid, attrs in [(nid, attrs) for nid, attrs in G.nodes(data=True) if attrs["type"] in [*ALL_SUBS, "ENV"]]:
            path = f"{db_base}/{attrs['type']}/{nid}"
            meta_path = f"{db_base}/metadata/{nid}"

            listener_paths["nodes"].append(path)
            listener_paths["meta"].append(meta_path)

        if edges is True:
            for src, trgt, attrs in G.edges(data=True):
                eid = attrs.get("id")
                epath = f"{db_base}/edges/{eid}"
                listener_paths["edges"].append(epath)

        return listener_paths

    def _fetch_g_data(self):
        print("Fetching entire graph data from Firebase RTDB")
        self.initial_data = {}

        paths = [
            f"{sub}"
            for sub in [*ALL_SUBS, "PIXEL", "ENV", "edges"]
        ]

        print("Fetch entire dir from FB")
        data = self.get_data(path=paths)
        if data:
            print(f"Data received from FB")
            return data



import dotenv
dotenv.load_dotenv()

if __name__ == "__main__":
    f = FirebaseRTDBManager()
    #f.delete_data(path="/")
    data = f.get_data(path=f"users/{os.environ.get('USER_ID')}/env/{os.environ.get('ENV_ID')}/")
    print("data", data)




