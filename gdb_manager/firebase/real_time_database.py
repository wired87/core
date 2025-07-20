import asyncio
import json
import os
import threading

import firebase_admin
import joblib
from firebase_admin import credentials
from firebase_admin import db
import logging # Good practice for backend applications

from dotenv import load_dotenv
from firebase_admin.db import Reference
from joblib import delayed

from qf_core_base.qf_utils.all_subs import ALL_SUBS

load_dotenv()


# DS PATH             fb_dest=f"users/{self.user_id}/datastore/{self.envc_id}/",
# G STATE PATH             fb_dest=f"users/{self.user_id}/env/{self.envc_id}/",
# GLOBAL STATE PATH f"{self.database}/global_states/"


# todo alle ds werden in gleichen apth geuppt (keine extra sessions) (vorerst)




class FirebaseRTDBManager:
    """
    A class to manage interactions with Firebase Realtime Database
    from a Python backend using the Admin SDK.

    Schemas:
    users/user_id/env/env_id/objects/qf/qfn_ids
    """

    def __init__(self, base_path, database_url: str or None = None):
        """
        Initializes the Firebase Admin SDK and gets a database reference.

        Args:
            service_account_key_path: Path to your Firebase Service Account JSON key file.
            database_url: The URL of your Firebase Realtime Database (e.g., 'https://YOUR-PROJECT-ID-default-rtdb.europe-west1.firebaseio.com/').
        """

        self.db_url = os.environ.get("FIREBASE_RTDB")

        print("Firebase url:", self.db_url)

        self._set_creds()

        try:
            if not firebase_admin._apps:
                 firebase_admin.initialize_app(self.creds, {
                    'databaseURL': self.db_url
                 })
                 logging.info("Firebase Admin SDK initialized successfully.")

            self.root_ref = db.reference(base_path)
            logging.info(f"Set FB root ref:{base_path}")

        except FileNotFoundError:
            logging.error(f"Service Account Key file not found.")
            raise
        except Exception as e:
            logging.error(f"Failed to initialize Firebase Admin SDK: {e}")
            raise

    def _get_ref(self, path: str):
        """Helper to get a database reference for a specific path."""
        if not path or path == '/':
            return self.root_ref
        # Ensure path starts without a leading slash if concatenating
        if path.startswith('/'):
             path = path[1:]
        return self.root_ref.child(path)



    def _set_creds(self):
        fb_creds = os.environ.get("FIREBASE_CREDENTIALS")
        if "{" in fb_creds:  # -> stringified fetched creds
            fb_creds = json.loads(fb_creds)

        self.creds = credentials.Certificate(
            fb_creds
        )


    def upsert_data(self, path: str, data: dict):
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
            self.root_ref.update(data) # or set
            logging.info(f"Successfully upserted data at path: {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to upsert data at path {path}: {e}")
            return False

    def upsert_batch(self, data, fb_dest=None):
        if fb_dest is not None:
            ref = db.reference(fb_dest)
        else:
            ref = self.root_ref
        try:
            print("ref")
            ref.update(data)
            logging.info(f"Successfully upserted data")
            #time.sleep(10)
            return True
        except Exception as e:
            logging.error(f"Failed to upsert data: {e}")
            return False


    def push_list_item(self, path, item):
        ref = db.reference(path)
        ref.push(item)
        print(f"Neues Element erfolgreich hinzugefügt unter Schlüssel: {path}")

    def get_init_env_tree(self, g, path):
        print("Data received")
        # load the graph with data
        env_attrs = None
        env_id = None
        data = self.get_data(path=path)
        if data:
            for node_type, node_id in data.items():
                for nid, attrs in node_id.items():
                    if node_type == "edges":
                        parts = nid.split("_")
                        g.add_edge(
                            parts[0],
                            parts[1],
                            attrs=attrs
                        )
                    elif node_type == "ENV":
                        env_attrs = attrs
                        env_id = nid
                    else:
                        g.add_node(
                            dict(
                                id=nid,
                                **attrs
                            )
                        )
        return g, env_id, env_attrs


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
            ref = db.reference("/")
            ref.delete()
            logging.info(f"Successfully deleted data at path: {path}")
            return True
        except Exception as e:
            logging.error(f"Failed to delete data at path {path}: {e}")
            return False

    def get_data(self, path: str):
        """
        Retrieves data from the specified path.

        Args:
            path: The path in the database (e.g., '/users/alice').
        Returns:
            The data at the path (as a dictionary or other Python type)
            or None if the path does not exist or on failure.
        """
        try:
            ref:Reference = self._get_ref(path)
            data = ref.get(path)
            if data is not None:
                logging.info(f"Successfully retrieved data from path: {ref._pathurl}:")
            else:
                 logging.info(f"No data found at path: {path}")
            #pprint.pp(data)
            return data
        except Exception as e:
            logging.error(f"Failed to retrieve data from path {path}: {e}")
            return None

    def start_listener_thread(
            self,
            db_path: list[str] or str,
            update_def,
            loop: asyncio.AbstractEventLoop or None = None,
        ):
        # Listen to changes in firebase
        self.listener_thread = threading.Thread(
            target=self._run_firebase_listener,
            args=(db_path, update_def, loop),  # Übergabe des Pfades und des Event Loops
            name=f"FBListener-{self.db_url}",
            daemon=True  # Der Listener-Thread wird beendet, wenn der Hauptprozess endet
        )
        self.listener_thread.start()

    def _run_firebase_listener(self, db_path: str or list[str], update_def, loop: asyncio.AbstractEventLoop or None = None, listener_type="db_changes"): # loop: asyncio.AbstractEventLoop,
        """
        Startet den blockierenden Firebase Realtime Database Listener.
        Läuft in einem separaten Thread.

        Args:
            db_path: Der Pfad in der Datenbank, auf den gelauscht werden soll.
            loop: Eine Referenz auf den asyncio Event Loop des Consumers.
        """

        if isinstance(db_path, str):
            db_path = [db_path]

        print(f"Listener Thread {threading.current_thread().name}: Starte Listener für {len(db_path)} Pfade (0:{db_path[0]}")

        try:
            def on_data_change(event):
                print(
                    f"Datenänderung empfangen: {event.event_type} - {event.path}: Listener Thread {threading.current_thread().name}")

                # Stellen Sie sicher, dass die Daten nicht None sind und verarbeiten Sie sie
                if event.data is not None:

                    update_payload = {
                        "type": listener_type,  # Oder ein anderer Typ für Updates
                        "path": event.path,  # Der spezifische Pfad der Änderung
                        "data": event.data  # Die geänderten Daten an diesem Pfad
                    }
                    # todo use
                    if loop is not None:
                        loop.call_soon_threadsafe(
                            asyncio.create_task,  # Erstellt eine Task im Event Loop
                            update_def(update_payload)  # Die Coroutine, die ausgeführt wird
                        )
                    else:
                        joblib.Parallel(n_jobs=1)(delayed(update_def()(update_payload)))

                else:
                    print(
                        f"Listener Thread {threading.current_thread().name}: Datenänderung empfangen: Daten sind None.")

            # Start listening
            for path in db_path:
                db.reference(path).listen(on_data_change)

            print(f"Listener Thread {threading.current_thread().name}: Listener für Pfad {db_path} beendet.")
        except Exception as e:
            print(f"Listener Thread {threading.current_thread().name}: FEHLER im Listener: {e}")


    def upsert_firebase(
            self,
            G,
            fb_dest=None,
            testing=False,
    ):

        if testing is False:
            updates = {
                f"{attrs.get('type')}/{nid}":
                    {k: v for k, v in attrs.items()}
                for nid, attrs in G.nodes(data=True) if attrs.get("type") not in ["USERS"]
            }

            for src, trgt in G.edges():
                edge_attrs = G[src][trgt]
                #print("edge_attrs", edge_attrs)
                for key,value in edge_attrs.items():
                    #print("Edge value", value)

                    path = f"edges/{src}_{value.get('rel')}_{trgt}"
                    updates.update(
                        {
                            path: {k: v for k, v in value.items() if k not in ["id", "symbol"]}
                        }
                    )
            # print("updates", updates)
        else:
            updates = {
                f"{attrs.get('type')}/{nid}": {k: v for k, v in attrs.items()}
                for nid, attrs in G.nodes(data=True) if attrs.get("type") not in ["USERS"]
            }

            for src, trgt in G.edges():
                edge_attrs = G[src][trgt]
                print("edge_attrs", edge_attrs)

                path = f"edges/{src}_{edge_attrs.get('rel')}_{trgt}"
                updates.update(
                    {
                        path: {k: v for k, v in edge_attrs.items() if k not in ["symbol"]}
                    }
                )
            # print("updates", updates)
        self.upsert_batch(updates, fb_dest)


    def get_listener_endpoints(self, nodes:list[str], metadata=False):
        return [
            f"{self.db_url}/{nid}" + "/metadata" if metadata is True else None
            for nid in nodes
        ]

    def _get_db_paths_from_G(self, G, db_base, metadata=False):
        # get paths for each node to lsiten to
        paths = []
        for nid, attrs in [(nid, attrs) for nid, attrs in G.nodes(data=True) if attrs["type"] in ALL_SUBS]:
            path = f"{db_base}/{attrs['type']}/{nid}"
            if metadata is True:
                path += "/metadata"
            paths.append(path)
        return paths





if __name__ == "__main__":
    f = FirebaseRTDBManager("")
    f.delete_data(path="/")



