import numpy as np

ALL_NP_LIBS=[f"np.{name}" for name in dir(np)
                if callable(getattr(np, name)) and not name.startswith('_')]