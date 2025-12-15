import time
from typing import Any, Callable
import ray

from app_utils import RUNNABLE_MODULES, ENVC, DIM, SHIFT_DIRS
import jax.numpy as jnp

from data import GAUGE_FIELDS
from module_manager.create_runnable import create_runnable
from utils._jax.jutils import JUtils
from utils._np.serialize_complex import serialize_arrays_from_dict

class DataUtils(
    JUtils
):

    """
    Creates Data (on CPU)
    Handles injections
    """

    def __init__(
            self, ntype, qfu, amount_nodes, world_cfg):
        JUtils.__init__(self)
        self.ntype=ntype
        self.amount_nodes=world_cfg["amount_nodes"]

        self.ref_tower = [
            (i for _ in range(DIM))
            for i in range(len(self.amount_nodes))
        ]

        self.qfu=qfu
        self.amount_nodes=amount_nodes
        self.soa:dict[str,list]={
            # add const
            **GAUGE_FIELDS[ntype]
        }
        self.attrs_struct: dict[str, dict] or None = None
        self.sum_util_keys: list[str] or None = None

        self.eq_utils = {}

    def print_payload(self,param_map):
        try:
            print(f"fire payload {self.ntype} SOA KEYS: {list(self.soa.keys())} --- {[f'{list(self.soa.keys())[p]}::{p}' for p in param_map]}")
        except Exception as e:
            print("Err print_payload", e)




    def create_attrs_tree(
            self
    ):
        """
        Create soa from args for entire world
        (field specific)
        format:
        {key:[[xv,],[yv,],[zv,],...]}

        -> upsert data Sp -> we have everything locally
        -> message
        """
        # todo speichere nur den feldwert + enrgie etc aber keine ableitungen-> die gehen direkt uins modelltraining
        print(f"{self.ntype} horizontal_to_vertical")
        self.soa = {}
        self.array_index_map = {}

        start = time.perf_counter_ns()

        # !WICHTIG: daten auf cpu erstellen da auf diesem weg env dynamisch -> lade von spanner!!!
        # todo : wie erhalte ich in jax.roll bei einem shift von 0,0,1 einen anderen index (ausgewählt)
        try:
            #for i in range(self.amount_nodes):
            self.soa = self.qfu.batch_field(
                ntype=self.ntype,
            )

        except Exception as e:
            print(f"Err horizontal_to_vertical {self.ntype}: {e}  ")

        end = time.perf_counter_ns()
        print("end create_attrs_tree", end-start)


    def grabber(self, index:list[int]):
        # convert 1d attrs struct to DIM
        self.index_map = index
        payload = jnp.array(
            [
                item[i]
                for i in index
            ] for item in list(self.soa.values())
        )
        print(f"grabbed items from {self.ntype}: {payload.shape}")
        return ray.put(payload)








    def convert_entries_ndim(self):
        # DIM wird als Konstante außerhalb der Methode angenommen
        converted = []
        # data_array = list of ntype param values for all points
        for k, data_array in self.soa.items():
            # Länge der gesamten Daten
            total_len = len(data_array)
            data_array = jnp.array(data_array)
            try:
                # jnp.split gibt eine Liste der neuen, kleineren Arrays zurück
                converted.append(jnp.split(data_array, DIM, axis=0))
            except Exception as e:
                print(f"Err splitting data for {k}: Array length {total_len} is not divisible by DIM ({DIM}):", e)
            return converted


    def print_state_soa(self):
        print(f"{self.ntype} soa: {', '.join([f'{key}: {type(value)}' for key, value in self.soa.items()])}")


    def create_eq_struct(
            self
    ) -> tuple[list, list]:
        print(f"create_eq_struct {self.ntype}")

        eq_struct: list[
            list[
                list[int or Any],
                Callable,
                int
            ]
        ] = []

        for item in self.arsenal_struct:
            try:
                print("working arsenal", item["nid"])

                param_map, axis_def = self.get_param_map(item)

                callable = create_runnable(
                    eq_code=item["code"],
                    eq_key=item["nid"],
                    xtrn_mods=RUNNABLE_MODULES
                )

                return_index: int = list(
                    self.soa.keys()
                ).index(
                    item["nid"]
                )

                # DEFINE PATHWAY PATTERN
                eq_struct.append([
                    param_map,
                    callable,
                    return_index,
                    axis_def
                ])

            except Exception as e:
                print("Err create_eq_struct", e, item, "self.soa.keys():", self.soa.keys())
        print(f"{self.ntype} eq_struct")

        self.eq_struct = eq_struct
        print("EQ struct created")


    def get_next_eq(self, eq_index):
        try:
            next_eq = self.arsenal_struct[eq_index + 1]
            return next_eq
        except Exception as e:
            print(f"Err get next eq: {e}")


    def extract_utils(self):
        """
        Extend attrs with utils
        """
        try:
            for attrs in self.attrs_struct:
                for key in self.sum_util_keys:
                    if key not in attrs:
                        val = self.ruc.get_step_eq_val(
                            attrs,
                            key,
                        )
                        attrs[key] = val
        except Exception as e:
            print(f"Err extract_utils: {e}")

    def vertical_to_horizontal(self):
        # scale params vertical -
        for i, attrs in enumerate(self.attrs_struct):
            for k, v in self.soa.items():
                attrs[k] = v[i]

        # serialize
        for i, attrs in enumerate(self.attrs_struct):
            serialize_arrays_from_dict(attrs)



    def check_key(self, key):
        """
        Set iter behaviour for params (const etc)
        """
        if key in ENVC:
            return None

        # For all other parameters, we map along the first axis.
        return 0



    def get_param_map(self, item):
        # create mapping pattern to parameters
        # include axis def for vmap

        param_map = []
        axis_def = []
        try:

            if len(item["params"]):
                for param in item["params"]:
                    # get list(index) of data package (on gpu) to extract
                    param_map.append(
                        list(
                            self.soa.keys()
                        ).index(param)
                    )

                    # set axes loop behaviour
                    axis_def.append(
                        self.check_key(param)
                    )

        except Exception as e:
            print(f"Err {self.ntype} get_param_map:", e, "self.soa.keys()", self.soa.keys(), "item params",
                  item["params"]
                  )
        return param_map, tuple(axis_def)



