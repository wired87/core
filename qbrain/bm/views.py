"""
BM views: health check and other HTTP endpoints.
"""
import os

from django.conf import settings
from django.http import FileResponse, Http404, JsonResponse


def health(request):
    """
    Health check endpoint for load balancers and orchestration (e.g. Cloud Run, K8s).
    Returns 200 if DB is reachable, 503 otherwise.
    """
    try:
        from qbrain.core.qbrain_manager import get_qbrain_table_manager
        qb = get_qbrain_table_manager()
        qb.run_query("SELECT 1", conv_to_dict=False)
        return JsonResponse({"status": "ok", "db": "connected"}, status=200)
    except Exception as e:
        return JsonResponse(
            {"status": "error", "db": "disconnected", "msg": str(e)},
            status=503,
        )


def spa_index(request):
    """
    Serve qdash SPA index.html for unmatched routes (React client-side routing).
    """
    index_path = settings.STATIC_ROOT / "index.html"
    if not index_path.exists():
        raise Http404("SPA index not found")
    return FileResponse(open(index_path, "rb"), content_type="text/html")
