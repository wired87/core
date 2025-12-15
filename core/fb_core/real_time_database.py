import json
import os
import re

import firebase_admin
from firebase_admin import db
import logging
from firebase_admin import credentials as fb_creds

from dotenv import load_dotenv
from firebase_admin.db import Reference

from app_utils import ENV_ID, USER_ID
from fb_core import FIREBASE_CREDS
from qf_utils.all_subs import ALL_SUBS

load_dotenv()
DB_URL = os.environ.get("FIREBASE_RTDB")

# todo alle ds werden in gleichen apth geuppt (keine extra sessions) (vorerst)

class FBRTDBMgr:
    """
    A class to manage interactions with Firebase Realtime Database
    from a Python backend using the Admin SDK.

    Schemas:
    users/user_id/env/env_id/objects/qf/qfn_ids
    """

    def __init__(self):
        """
        Initializes the Firebase Admin SDK and gets a database reference.

            database_url: The URL of your Firebase Realtime Database (e.g., 'https://YOUR-PROJECT-ID-default-rtdb.europe-west1.firebaseio.com/').
        """
        with open(FIREBASE_CREDS, "r", encoding="utf-8") as f:
            _creds = json.load(f)
        self.creds = fb_creds.Certificate(_creds)
        self.db_url = DB_URL

        print("Firebase url:", self.db_url)

        if not firebase_admin._apps:
            firebase_admin.initialize_app(self.creds, {
                'databaseURL': DB_URL
            })
            print("Firebase Admin SDK initialized successfully.")

        self.invalid_keys_detected = []

    def set_root_ref(self, base_path):
        if isinstance(base_path, str):
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

        if list_entry is True:
            db.reference(path).push(data)
        else:
            db.reference(path).update(data)
        return True


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

    def get_data(self, path: str or list, child=True):
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

                data = ref.get()
                if data is not None:
                    print(f"Successfully retrieved data from path: {ref._pathurl}:")
                else:
                     raise ValueError(f"No data found at path: {path}")

                if isinstance(data, tuple):
                    print("RECEIVED DATA AS TUPLE ")
                    data = data[0]
                sub_data[p] = data

            sub_data:dict = self.filter_raw_graph_data_keys(sub_data)
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

                ##pprint.pp(update_item)
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

        self.upsert_data(
            fb_dest,
            data=updates,
            list_entry=datastore
        )


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

    def _get_db_paths_from_G(self, g, db_base, edges=True):
        # get paths for each node to lsiten to
        listener_paths = {
            "nodes": [],
            "EDGES": [],
            "meta": [],
            "global": [],
            "value": []
        }

        listener_paths["global"].append(
            f"{db_base}/global_states/"
        )

        for nid, attrs in [(nid, attrs) for nid, attrs in g.G.nodes(data=True) if attrs["type"] in [*ALL_SUBS, "ENV"]]:
            ntype = attrs['type']

            path = f"{db_base}/{ntype}/{nid}"
            meta_path = f"{db_base}/metadata/{nid}"
            value_path = f"{db_base}/{ntype}/{nid}/{g.qf_utils._field_value(ntype)}"

            listener_paths["nodes"].append(path)
            listener_paths["meta"].append(meta_path)
            listener_paths["value"].append(value_path)

        if edges is True:
            for src, trgt, attrs in g.G.edges(data=True):
                eid = attrs.get("id")
                epath = f"{db_base}/edges/{eid}"
                listener_paths["EDGES"].append(epath)

        return listener_paths

    def _fetch_g_data(self, db_root):
        print("Fetching entire graph data from Firebase RTDB")
        self.initial_data = {}

        # Create paths
        paths = [
            f"{db_root}/{sub}"
            for sub in [*ALL_SUBS, "PIXEL", "ENV", "EDGES"]
        ]

        print("Fetch entire dir from FB")
        data = self.get_data(path=paths)
        if data:
            print(f"Data received from FB")
            return data
        print("no data could be fetched")


    def filter_raw_graph_data_keys(
            self, data
    ) -> dict:
        converted_data = {}
        for k, v in data.items():
            if k.endswith("/"):
                # rm last slash
                k = k[:-1]
            new_key = k.split("/")[-1]
            print(f"Converted {k} -> {new_key}")
            converted_data[new_key] = v
        return converted_data




if __name__ == "__main__":
    f = FBRTDBMgr()
    #f.delete_data(path="/")
    path = f"users/{USER_ID}/env/{ENV_ID}/cfg/{ENV_ID}/world/"
    data = f.get_data(path=path)
    print("data")
    ##pprint.pp(data)



