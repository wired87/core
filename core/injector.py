import random

try:
    import ray
except ImportError:
    class MockRay:
        def remote(self, *args, **kwargs):
            if len(args) == 1 and hasattr(args[0], "__call__"):
                 return args[0]
            return lambda cls: cls
    ray = MockRay()

from _ray_core.base.base import BaseActor
from core.app_utils import FB_DB_ROOT, TESTING
from fb_core.real_time_database import FBRTDBMgr
from qf_utils.all_subs import ALL_SUBS

from qf_utils.field_utils import FieldUtils
from qf_utils.qf_utils import QFUtils
from graph.local_graph_utils import GUtils


@ray.remote
class Injector(
    BaseActor,
    FieldUtils,
):

    """
    fetch and save ncfg
    Todo later come back to blcoks and phase. for now keep jus blocks
    """
    def __init__(
            self,
            world_cfg
    ):
        BaseActor.__init__(self)
        FieldUtils.__init__(self)
        self.ncfg = {}
        self.world_cfg = world_cfg
        self.amount_nodes = world_cfg["amount_nodes"]
        self.firebase = FBRTDBMgr()

        self.sim_time = world_cfg["sim_time"]

        self.cluster_schema = []

        self.schema = [
            (0 for _ in self.dim)
            for _ in range(self.amount_nodes)
        ]
        self.injector_schema:dict[str, dict] = self.christmas_tree()



    def christmas_tree(self):
        return  {
            sub:{
                t: [
                    list(random.choices(self.schema, 5)),
                    [
                        random.randint(0, 10)
                        for _ in range(5)
                    ]
                ]
                for t in range(self.sim_time)
            }
            for sub in ALL_SUBS # todo change dynamic get from G
        }

    def get_injector(self, t, ntype):
        return self.injector_schema[ntype][t]


    def ncfg_by_time(self, time:int):

        """
        This method imlices all ndim param values are merged to a big list

        punkt : param : [[t], [v]]
        """

        applier_struct = []
        for ntype, struct in self.ncfg.items():
            # add ntype dim
            applier_struct.append([])

            for j, item in enumerate(struct):
                # item[j] represents parameter index
                time_series: list[int] = item[0]
                strenght_series: list[int] = item[1]

                if time in time_series or self.check_repeatment():
                    index = time_series.index(time)
                    strength = strenght_series[index]

    def set_inj_pattern(
        self,
        inj_struct: dict[
            str,
            list[tuple[tuple[int],
            list[list[int], list[int]]
            ]]
        ]) -> list[
        int,  # module id
        tuple[
            tuple,  # pos
            float,  # e
        ]
    ]:
        # ganzer cpu processing stuff wird relay frontend:
        # pi:t:e
        #
        # tod later make docker image for gpu and cpu!

        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE"
        )

        self.module_pattern_collector = []

        for i, module in enumerate(modules):
            mid = module["id"]
            keys: list[str] = module["keys"]
            faxis_def: list[int or None] = module["axis_def"]

            # get module fields
            fields = self.g.get_neighbor_list(
                node=mid,
                target_type="FIELD",
            )
            new_struct = [
                [] for _ in range(len(modules))
            ]
            for j, (fid, fattrs) in fields:
                if fid in inj_struct:
                    for pos, inj_struct in inj_struct[fid]:
                        new_struct[
                            i
                        ].append(
                            [
                                j,

                            ]
                        )
        pass

        

    def get_update_rcv_ncfg(
            self,
            attr_struct:list[dict]
    ):

        self.g = GUtils(
            G=self.get_G(),
        )

        self.qfu = QFUtils(self.g)

        if not len(list(self.ncfg.keys())):
            self.get_ncfg()

        # prod
        updated_attr_struct = self.apply_stim_attr_struct(
            attr_struct
        )
        print("Finished injection process")
        return updated_attr_struct


    def get_ncfg(self):
        print("===============NODE CFG PROCESS=================")
        try:
            if TESTING is False:
                # GET ALL NFCGs
                ncfg_path = f"{FB_DB_ROOT}/cfg/node/"
                ncfg: dict[
                    str,
                    list[tuple[tuple[int],
                    list[list[int], list[int]]
                    ]]
                ] = self.firebase.get_data(
                    path=[ncfg_path]
                )
                print(f"NCFG received: {ncfg}")
                if ncfg is not None and "node" in ncfg:
                    ncfg = ncfg["node"]
            else:
                ncfg = {}

            for nid, ncfg_item in ncfg.items():
                # check and convert nid to new format
                if not "__" in nid and "_px" in nid:
                    nid = nid.replace("_px", "__px")

                self.ncfg[nid] = ncfg_item
            print("NCFG process finsihed successfully")

            id_list = list(self.ncfg.keys())
            print("NCFG id_list", id_list)

            # Create nodes for all specified fields in ncfg ->
            # save in G
            print("finished node_cfg_process")
            return id_list
        except Exception as e:
            print(f"Err node_cfg_process: {e}")


    def apply_stim_default(
            self,
            attrs_struct:list[dict]
    ):
        if len(attrs_struct):
            current_iter = attrs_struct[0]["tid"]
            if current_iter // self.world_cfg["phase"] == 0:
                for attrs in attrs_struct:
                    if attrs.get("type") == self.world_cfg["particle"]:
                        attrs["energy"] = self.world_cfg["energy"]
                    #print("Stim applied")
            else:
                print("Skipping stim at iter", current_iter)
            return attrs_struct
        else:
            print("apply_stim_default has no len attrs_struct")

    def apply_stim_attr_struct(
            self,
            attr_struct:list[dict]
    ):
        """

        Checks each node of a list for current stim phase and applies energy to it

        """
        # gnn kannit ur einem node alles ancheinander prozessieren
        print("Applying stim to attr struct")

        for attrs in attr_struct: # loop each field attrs
            try:
                nid = attrs["id"]
                tid = attrs["tid"]
                if nid in self.ncfg: # do we have a ncfg for this node?
                    total_iters = self.ncfg[nid].get("total_iters", 0)
                    print("total_iters", total_iters)
                    if total_iters == 0:
                        blocks:list = self.ncfg[nid]["blocks"]
                        print("blocks", blocks)
                        for block in blocks: # loop blocks
                            for phase in block: # loop single phase
                                total_iters += int(phase["iters"])
                                print("new total_iters", total_iters)

                    # calculate the rest of t_i / tid
                    rest_val: int = total_iters // tid
                    print("rest_val", rest_val)

                    # get current phase
                    total_phase_iters = 0
                    current_phase = None
                    for block in self.ncfg[nid]["blocks"]:
                        print("block", block)
                        for phase in block:
                            print("phase", phase)

                            total_phase_iters += int(phase["iters"])
                            if total_phase_iters > rest_val:
                                # Right phase found
                                current_phase=phase
                                print("new current_phase", current_phase)
                                break

                    if current_phase is not None:
                        attrs["energy"] = current_phase["energy"]
                        print("Applied stim to", nid, attrs["energy"])
                    else:
                        print("Err: couldnt identify aphase to apply stim to")
            except:
                print(f"Err: couldnt identify aphase: {nid}")
        return attr_struct

