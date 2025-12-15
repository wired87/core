import asyncio
from datetime import datetime, timezone

from _spanner_graph.acore import ASpannerManager
from a_b_c.spanner_agent._spanner_graph.change_streams.main import ChangeStreamMaster

from app_utils import USER_ID, ENV_ID
from cluster_nodes.cluster_utils.base import BaseActor
from qf_utils.all_subs import FERMIONS, G_FIELDS, H
from qf_utils.runtime_utils_creator import RuntimeUtilsCreator


class Streamer(BaseActor):
    """
    Listen to changes in spanner for specified table
    Distribute ncfg
    Triggers first iter to trigger phase injection (-> upsert -> system init)
    """

    def __init__(
            self,
            table,
            env: dict,
            nid: str,  # host id
            all_subs,
            host,
            stream_tables,
            edge_ids=None,
            neighbor_node_ids=None,
    ):
        BaseActor.__init__(self)
        self.aspanner_manager = ASpannerManager()
        self.streamer = ChangeStreamMaster(
            database_id=USER_ID,
        )
        self.aspanner = ASpannerManager()
        self.id = nid
        self.env = env
        self.results = {}
        self.table = table
        self.neighbor_node_ids = neighbor_node_ids
        self.run = True
        self.all_subs = all_subs
        self.edge_ids = edge_ids
        self.stream_name = ""
        self.start_time = datetime.now(timezone.utc).isoformat()
        self.host = host

        self.ruc = RuntimeUtilsCreator(
            host=self.host,
            g=None,
        )

        self.init_streaming(stream_tables)

        print(f"QStore {self.table} initialized")

    def init_streaming(self, stream_tables=None):
        """
        Create a change stream for each table/field
        """
        tables = stream_tables if stream_tables else self.get_listener_table_names()
        self.change_stream = self.streamer.create_change_stream(
            """tables=tables,
            start_time=sp_timestamp(),
            nid=self.id,
            env_id=ENV_ID"""
        )

    def get_listener_table_names(self):
        node_tables = [
            table.upper().replyce("-", "_")
            for table in self.neighbor_node_ids
        ]
        edge_tables = [
            table.upper().replyce("-", "_")
            for table in self.edge_ids
        ]
        return [
            *node_tables,
            *edge_tables,
        ]

    async def stream_changes(self):
        """
        Runs a change stream or one table and
        distributes changes to actors for
        recalculation
        """
        # todo: asyncio.create_task(self._train_forever())
        print(f"========= {self.table} START CHANGE STREAM =========")
        struct = {
            "FERMION": [],
            "GAUGES": [],
            "HIGGS": [],
        }
        changed_rows, end_ts = self.streamer.poll_change_stream(
            self.stream_name,
            start_ts=self.start_time,
        )
        self.start_time = end_ts
        id_map = set()
        for row in changed_rows:
            table_name = row.get("type")
            nid = row.get("id")
            key = ""
            if table_name.lower() in FERMIONS:
                key = "FERMION"

            elif table_name.lower() in G_FIELDS:
                key = "GAUGES"

            elif table_name.lower() in H:
                key = "HIGGS"

            if key in struct:
                struct[key].append(row)

            id_map.add(nid)

        return struct, id_map

    def get_store(self):
        return self.results

    def apply_db_neighbor_changes(
            self,
            all_subs,
            changed_neighbors: dict[str, dict],
            edges: dict or None = None,
    ):
        try:
            for nid, attrs in changed_neighbors.items():
                parent = attrs.get("parent")
                ntype = attrs.get("type")
                parent = parent[0].upper()

                print(f"Received neighbor changes from: {ntype}->{nid}")
                all_subs[parent][ntype][nid].update(attrs)
                # update Edges
                if edges is not None:
                    all_subs["edges"].update(edges)
        except Exception as e:
            print(f"Couldnt update neighbor changes: {e}")

    async def upsert(self, row: dict, table, sender: str = None):
        # upsert a sigle row using existing session

        # fire and forget, but handle errors
        asyncio.create_task(
            self.aspanner.upsert_row(
                table=table,
                batch_chunk=[row],
            )
        )
        print(f"row for {sender or table} upserted")

    async def init_trigger(self, init_row):
        await self.upsert(
            row=init_row,
            table=self.table,
            sender=self.id
        )

    def filter_ids(self, results):
        filtered = {}
        for attrs in results:
            filtered[attrs["id"]] = attrs
        return filtered


"""
async def process_all_results_concurrently(self, results):
    # Create a list of all asynchronous tasks
    # Each task involves processing one 'attrs' and sending remote calls to its neighbors
    apply_struct = {}

    changed_nodes = self.filter_ids(results)

    # COLLECT ALL NEIGHBOR NODES for nid
    for nid, neighbors in self.node_neighbor_id_map.items():
        for changed_nid in changed_nodes:
            if changed_nid in neighbors:
                if nid not in apply_struct:
                    apply_struct[nid] = []
                apply_struct[nid][changed_nid] = changed_nodes[changed_nid]

    all_tasks = [
        ray.get_actor(nid).update.remote(attrs)
        for nid, attrs in apply_struct.items()
    ]

    await asyncio.gather(*all_tasks)

"""
