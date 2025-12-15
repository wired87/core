import jax
from typing import List, Any


def count_gpus() -> float:
    """
    Retrieves the count of available GPU devices and returns it as a float.
    """

    # 1. Get the list of all available devices managed by JAX
    devices: List[Any] = jax.devices()

    # 2. Filter for GPU devices
    # We count only devices where the device type attribute is 'gpu'
    gpu_count: int = sum(1 for device in devices if device.device_kind == 'gpu')
    print(f"Available GPUs: {gpu_count}")

    # 3. Convert the integer count to a float
    return float(gpu_count)

