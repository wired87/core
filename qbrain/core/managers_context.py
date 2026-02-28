"""
Context for orchestrator-provided managers. When orchestrator dispatches handlers,
it sets itself as the current orchestrator so handlers can access its managers.
"""
from contextvars import ContextVar
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from qbrain.core.orchestrator_manager.orchestrator import OrchestratorManager

_orchestrator_context: ContextVar[Optional["OrchestratorManager"]] = ContextVar(
    "orchestrator_context", default=None
)


def set_orchestrator(orch: "OrchestratorManager"):
    """Set the current orchestrator for this context."""
    return _orchestrator_context.set(orch)


def reset_orchestrator(token):
    """Reset the orchestrator context."""
    _orchestrator_context.reset(token)


def get_orchestrator() -> Optional["OrchestratorManager"]:
    """Return the current orchestrator, or None if not in dispatch context."""
    return _orchestrator_context.get()


def get_env_manager():
    """Return env_manager from current orchestrator, or default."""
    orch = get_orchestrator()
    if orch is not None:
        return orch.env_manager
    from qbrain.core.env_manager.env_lib import _default_env_manager
    return _default_env_manager


def get_session_manager():
    """Return session_manager from current orchestrator, or default."""
    orch = get_orchestrator()
    if orch is not None:
        return orch.session_manager
    from qbrain.core.session_manager.session import _default_session_manager
    return _default_session_manager


def get_injection_manager():
    """Return injection_manager from current orchestrator, or default."""
    orch = get_orchestrator()
    if orch is not None:
        return orch.injection_manager
    from qbrain.core.injection_manager.injection import _default_injection_manager
    return _default_injection_manager


def get_file_manager():
    """Return file_manager from current orchestrator, or default."""
    orch = get_orchestrator()
    if orch is not None:
        return orch.file_manager
    from qbrain.core.file_manager.file_lib import _default_file_manager
    return _default_file_manager


def get_model_manager():
    """Return model_manager from current orchestrator, or default."""
    orch = get_orchestrator()
    if orch is not None:
        return orch.model_manager
    from qbrain.core.model_manager.model_lib import _default_model_manager
    return _default_model_manager


def get_param_manager():
    """Return param_manager from current orchestrator, or default."""
    orch = get_orchestrator()
    if orch is not None:
        return orch.param_manager
    from qbrain.core.param_manager.params_lib import _default_param_manager
    return _default_param_manager


def get_module_manager():
    """Return module_manager from current orchestrator, or default."""
    orch = get_orchestrator()
    if orch is not None:
        return orch.module_db_manager
    from qbrain.core.module_manager.ws_modules_manager.modules_lib import _default_module_manager
    return _default_module_manager


def get_field_manager():
    """Return field_manager from current orchestrator, or default."""
    orch = get_orchestrator()
    if orch is not None:
        return orch.field_manager
    from qbrain.core.fields_manager.fields_lib import _default_field_manager
    return _default_field_manager


def get_method_manager():
    """Return method_manager from current orchestrator, or default."""
    orch = get_orchestrator()
    if orch is not None:
        return orch.method_manager
    from qbrain.core.method_manager.method_lib import _default_method_manager
    return _default_method_manager


def get_user_manager():
    """Return user_manager from current orchestrator, or default."""
    orch = get_orchestrator()
    if orch is not None:
        return orch.user_manager
    from qbrain.core.user_manager.user import _default_user_manager
    return _default_user_manager
