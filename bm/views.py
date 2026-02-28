"""
BM views: health check and other HTTP endpoints.
"""
import os

from django.http import JsonResponse


def health(request):
    """
    Health check endpoint for load balancers and orchestration (e.g. Cloud Run, K8s).
    Returns 200 if DB is reachable, 503 otherwise.
    """
    try:
        from core.qbrain_manager import get_qbrain_table_manager
        qb = get_qbrain_table_manager()
        qb.run_query("SELECT 1", conv_to_dict=False)
        return JsonResponse({"status": "ok", "db": "connected"}, status=200)
    except Exception as e:
        return JsonResponse(
            {"status": "error", "db": "disconnected", "msg": str(e)},
            status=503,
        )
