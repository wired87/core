import base64
import json
from fractions import Fraction
from typing import Any

import numpy as np
import jax.numpy as jnp
# save as list - convert in runtime
def serialize_complex(com):

    """

    Serialisiert oder deserialisiert ein beliebig verschachteltes Array oder Listenstruktur.

    """

    try:
        COMPLEX_TYPES = (complex, np.complex128)
        FLOAT_TYPES = (float, np.float64)
        INT_TYPES = (int, np.int64)
        ARRAY_TYPES = (np.ndarray, jnp.array)
        # --- 1. Skalare behandeln ---
        if isinstance(com, COMPLEX_TYPES):
            return [float(com.real), float(com.imag)]

        if isinstance(com, FLOAT_TYPES):
            # Bereinigung kritischer Floats (INF, NaN)
            if np.isinf(com) or np.isnan(com):
                return 0.0
            return float(com)

        if isinstance(com, INT_TYPES):
            return int(com)

        # --- 3. Listen/Tupel (Rekursion) behandeln ---
        if isinstance(com, (list, tuple)):
            # Rekursiv die Elemente der Liste/des Tupels serialisieren
            return [serialize_complex(item) for item in com]

        # --- 4. Wörterbücher (Dictionaries) behandeln ---
        if isinstance(com, ARRAY_TYPES):
            return [serialize_complex(item) for item in com.tolist()]

        if isinstance(com, str):
            return com

        if isinstance(com, dict):
            # Rekursiv Schlüssel-Wert-Paare serialisieren
            return {k: serialize_complex(v) for k, v in com.items()}
        raise ValueError(f"Unknown serialize type: {type(com)} for value {com}")
    except Exception as e:
        print("serialization err:", e)



def deserialize_bytes_dict(bytes_struct, from_json=True, key=None, **args):

    """
    Deserialisiert ein einzelnes oder verschachteltes serialisiertes Array.
    """

    try:
        # Falls String, erst JSON laden
        if from_json and isinstance(bytes_struct, str):
            bytes_struct = json.loads(bytes_struct)

        if isinstance(bytes_struct, dict):
            bytes_struct = bytes_struct["serialized_complex"]
        if (
                isinstance(bytes_struct, list)
                and len(bytes_struct) == 2
                and all(isinstance(x, (int, float)) for x in bytes_struct)
        ):
            #print(" len(bytes_struct) == 2 v  bytes_struct", bytes_struct)
            restored = np.complex128(complex(bytes_struct[0], bytes_struct[1])) #restored = np.complex128(complex(bytes_struct[0] + bytes_struct[1]))

        elif isinstance(bytes_struct, list) and isinstance(bytes_struct[0], list):
            restored = jnp.array([deserialize_bytes_dict(item) for item in bytes_struct])

        else:
            restored = bytes_struct
        #print(f"deserialized complex: {restored}")
        return restored
    except Exception as e:
        print("Error deserialize struct", e)


def check_serilisation(data):
    try:
        json.dumps(data)
        return data
    except Exception as e:
        serialized = serialize_complex(data)
        return serialized

def convert_numeric(v):
    try:
        return Fraction(v)
    except Exception as e:
        return v


def check_serialize_dict(data, attr_keys=None):
    try:
        new_dict={}
        for k, v in data.items():
            if attr_keys is not None:
                if k in attr_keys:
                    #print("Convert sd key:", k, type(v), v)
                    v = check_serilisation(v)
                    new_dict[k] = v
            else:
                # print("Convert sd key:", k, type(v), v)
                v = check_serilisation(v)
                new_dict[k] = v
        return new_dict
    except Exception as e:
        print("Error serialize dict", e)
        return data


def serialize_arrays_from_dict(data):
    for k, v in data.items():
        if not isinstance(
                v, (str, int, float, bool, type(None))
        ):
            try:
                new_stuff = serialize_complex(v)
                data[k]=new_stuff
            except Exception as e:
                print("Error serialize dict", e)
    return data

def serialize_to_bytes_dict(v):
    if isinstance(v, jnp.ndarray):
        try:
            a = np.asarray(v)
            shape = a.shape
            dtype = str(a.dtype)
            _bytes = base64.b64encode(np.nan_to_num(a).tobytes()).decode('utf-8')
            return {
                "shape": shape,
                "bytes": _bytes,
                "dtype": dtype,
            }
        except Exception as e:
            print("Error serialize dict", e, v)


def deserialize_dict(v:dict or Any):
    if isinstance(v, dict) and "bytes" in v:
        try:
            decoded_bytes = base64.b64decode(v["bytes"])

            # 2. Convert the bytes object to a 1D NumPy array using the original dtype
            # The function np.frombuffer reads the bytes and interprets them as an array
            np_array_1d = np.frombuffer(decoded_bytes, dtype=v["dtype"])

            # 3. Reshape the 1D NumPy array back to its original shape
            np_array_restored = np_array_1d.reshape(v["shape"])

            # 4. Convert the NumPy array to a JAX NumPy array (jnp.array)
            v = jnp.asarray(np_array_restored)
        except Exception as e:
            print("Error deserialize dict", e)
    else:
        #print("skip deserilizatio of ", v)
        pass
    return v


def restore_row(data:dict):
    new_dict = {}
    #print("restore_selfdict data B4:", data)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, dict) and "serialized_complex" in v:  # ("bytes" in v or "data" in v)
                v = deserialize_bytes_dict(
                    bytes_struct=v,
                )
            #v = convert_numeric(v)
            new_dict[k] = v
    return new_dict

