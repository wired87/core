"""
Optional ejkernel integration: use for attention/kernel ops inside step or loss.
See https://ejkernel.readthedocs.io and https://github.com/erfanzar/ejkernel.
If ejkernel is not installed, APIs here are no-ops or return None.
"""
from typing import Any, Optional

_ejkernel_available = False
try:
    import ejkernel  # noqa: F401
    _ejkernel_available = True
except ImportError:
    pass


def get_ejkernel_config() -> Optional[Any]:
    """
    Return ejkernel config if available; else None.
    Use inside step_fn / loss_fn to pass config to kernel/attention ops.
    Override or extend when using ejkernel (e.g. 7-tier config, backend selection).
    """
    if not _ejkernel_available:
        return None
    return {}


def is_ejkernel_available() -> bool:
    """True if ejkernel is installed and usable."""
    return _ejkernel_available
