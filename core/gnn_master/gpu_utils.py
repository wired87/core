import time

import jax
import jax.numpy as jnp

class GPUBaseActor:
    def __init__(self):
        super().__init__()

    def test_gpu_ready(self, device):
        print("🔍 Available devices:", jax.devices())

        # Device wählen (GPU, wenn vorhanden)
        print(f"✅ Using device: {device}")

        # Matrix erstellen
        x = jnp.ones((1000, 1000), dtype=jnp.float32)

        # Warmlauf (JAX jit-kompiliert beim ersten Run)
        _ = x @ x

        # Zeit messen
        start = time.time()
        y = x @ x
        jax.block_until_ready(y)   # zwingt GPU/CPU zur Ausführung, nicht lazy
        end = time.time()

        print("⏱️ Time:", round(end - start, 6), "seconds")
        print("Result shape:", y.shape, "example value:", y[0,0])
