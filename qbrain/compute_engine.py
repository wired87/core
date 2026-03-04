"""
Compatibility shim for older imports.

Some parts of qbrain (e.g. deployment workflows) expect `qbrain.compute_engine.VMMaster`
to exist. In this repo snapshot the full compute engine implementation is not present,
so we provide a minimal base class to keep imports working.
"""


class VMMaster:
    def __init__(self, project_id=None, zone=None, *args, **kwargs):
        self.project_id = project_id
        self.zone = zone

    def __getattr__(self, item):
        raise NotImplementedError(
            f"Compute engine functionality is not available (missing implementation). "
            f"Tried to access: {item}"
        )

