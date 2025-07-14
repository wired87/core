import asyncio
import time

import networkx as nx
import ray

from qf_core_base.qf_utils.all_subs import ALL_SUBS
from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER


class HeadMessageManager:

    """
    Head node message handler
    """

    def __init__(self, parent, attrs, user_id, env, database, G):
        self.attrs=attrs
        self.attrs_id= attrs.get("id")
        self.env=env
        self.database=database
        self.user_id=user_id

        self.head_ref = ray.get_actor(env["id"])

        self.parent = parent
        self.g = GUtils(
            nx_only=False,
            G=G,
            g_from_path=None,
            user_id=self.user_id,
        )
        self.ids_distributed = []

    async def _start(self):
        """
        loop through all specified nodes and
        """
        LOGGER.info(f"ENV _start request received")
        all_subs = await self.head_ref.get_all_subs.remote()
        # Start Agents
        start_time = int(time.time())
        await asyncio.gather(*[
            ref.receiver.receive.remote(
                data={
                    "type": "start",
                    "data": {
                        "start_time": start_time + 10,
                        "host": {
                            "id": self.attrs["id"]
                        }
                    }
                })
            for ref in [attrs["ref"] for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") in all_subs]
        ])

    async def _init_handshake(self, payload):
        """
        internal hs req from worker -> respond with attrs
        """
        final_attrs = {}
        final_nid = ""

        pod_type = payload["data"]["ntype"]
        ref = payload["data"]["ref"]

        # Set ALL_SUBS IN Head
        all_subs = payload["data"]["all_subs"]
        await self.head_ref.set_all_subs.remote(all_subs)

        for nid, attrs in self.g.G.nodes(data=True):
            ntype = attrs.get("type")
            if ntype.upper() == pod_type.upper() and nid not in self.ids_distributed:
                self.ids_distributed.append(nid)
                final_attrs = attrs
                final_nid = nid
                print("Matching attrs localized. break event loop.")
                break

        # Set ref in
        self.g.G.nodes["ref"] = ref

        # Give ref node a name (final_nid)
        ref.options(name=final_nid).remote()

        # prepare res data pckg
        response_payload = self._get_handshake_response_content(
            nid=final_nid,
            attrs=final_attrs
        )
        # Return attrs to
        return response_payload

    def _get_handshake_response_content(self, nid, attrs):
        "Get all qfn neighbors of"

        # todo erstelle tiny_G nur center (ref) node u direkte nachbarm = self._build_tinyG(nid)

        response_payload = dict(
            env=self.env,
            attrs=attrs,
            G=self.g.G,
            database=self.database,
            user_id=self.user_id
        )
        return response_payload


    def _build_tinyG(self, nid) -> dict:
        # Extract all sub_fields and qfns of direct neighbor from

        # get parent qfn
        qfn_parent_id, qfn_parent_attrs = self.g.get_single_neighbor_nx(nid, target_type="QFN")
        qfn_neighbors = self.g.get_neighbor_list(qfn_parent_id, target_type="QFN")

        # hs ref and build tinyG
        local_g_utils = GUtils(
            user_id=self.user_id
        )

        for qfnid, attrs in qfn_neighbors:
            local_g_utils.add_node(attrs=attrs)
            sub_fields = self.g.get_neighbor_list(qfnid, trgt_rel="has_field")
            for snid, snattrs in sub_fields:
                local_g_utils.add_node(attrs=snattrs)

        LOGGER.info(f"local_g_utils build for {nid}")

        # return serializable tG
        return nx.node_link_data(local_g_utils.G)




    # get paths of all nodes -> fb
    async def _stop(self):
        LOGGER.info(f"ENV stop request received")

        # each qfn finishes its iteration ->
        # pushes changes to neighbors
        # receives "stop" message push last updates
        sd_response = await self.stop_qfns()
        self.state = "inactive"
        return sd_response

    async def _stim_handler(self):
        # todo
        return True

    async def send_message(self, data):
        # Receive a intern child message of a qfn
        # if self.extenal_vm is True: must called
        # in ServerNode
        # here just save
        sub_type = data.get("sub_type")
        if sub_type == "state":
            pass

    async def _handle_db_change(self, payload):
        # payload = id:new_data
        # called when listener recognizes changes in DB
        # ws (für testing auf einem server)
        # todo sned to ServerWorker
        LOGGER.info(f"ENV _handle_db_change request received")
        filtered_payload = payload  # todo

    async def _state_req(self, payload):
        if payload == getattr(self.parent, "id"):
            # Frontend Updator status request
            return (self.attrs_id, self.state)
        else:
            LOGGER.info(f"Mismatch request:self host id: {payload}:{self.attrs['id']}")


    async def _worker_status(self, payload):
        state= payload["data"]["state"]
        nid= payload["data"]["id"]
        active_workers = await self.head_ref.get_ative_workers.remote()

        if state == "standby" and nid not in active_workers:
            self.head_ref.handle_active_worker_states.remote(
                "append",
                nid
            )

        elif state == "shutdown" and nid not in active_workers:
            self.head_ref.handle_active_worker_states.remote(
                "remove",
                nid
            )

        else:
            LOGGER.error(f"Unknown status request from node: {nid} status: {state}")



    async def stop_qfns(self):
        failed_shut_downs = []
        all_qfns = [(nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") in ALL_SUBS]

        async def _stop_request(nid, attrs):
            future = attrs["ref"].receiver.receive.remote(
                data={
                    "data": {
                        "host": {
                            "id": self.attrs["id"]
                        }
                    },
                    "type": "stop"
                })
            result = await ray.get(future)
            if isinstance(result, str):
                LOGGER.info(f"Shut down for node {nid} failed!")
                failed_shut_downs.append((nid, result))
            else:
                LOGGER.info(f"Node {nid} shutdown.")

        # stop all nodes -> nodes upsert by its own -> todo collect all by env -> better debugging
        await asyncio.gather(*[_stop_request(nid, attrs) for nid, attrs in all_qfns])

        LOGGER.info(f"{len(failed_shut_downs)}/{len(all_qfns)} shutdowns failed")

        if len(failed_shut_downs):
            return failed_shut_downs
        return True
