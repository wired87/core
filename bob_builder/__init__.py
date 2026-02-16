"""
bob_builder package

We re-export the main workflow entrypoint in a lazy way to avoid
importing `workflow` at package import time. This prevents the
`runpy` RuntimeWarning that appears when running:

    python -m bob_builder.workflow
"""


def build_and_deploy_workflow(*args, **kwargs):
    """Lazy proxy to `workflow.build_and_deploy_workflow`."""
    from .workflow import build_and_deploy_workflow as _impl

    return _impl(*args, **kwargs)

