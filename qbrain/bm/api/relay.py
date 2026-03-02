"""
Case-struct pickup for DRF: discover relay cases from RELAY_CASES_CONFIG and register
one API route per case. Production-ready pattern: single view, URL-driven dispatch.
"""
from django.urls import path
from rest_framework.views import APIView
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework import status

from qbrain.predefined_case import RELAY_CASES_CONFIG


def get_relay_cases():
    """Return list of case dicts that have a callable handler (func)."""
    cases = []
    for c in RELAY_CASES_CONFIG or []:
        if not isinstance(c, dict):
            continue
        case_name = c.get("case")
        func = c.get("func")
        if case_name and callable(func):
            cases.append(c)
    return cases


def _build_auth(request: Request, body: dict) -> dict:
    """Merge auth from request (user, headers) and body."""
    auth = dict(body.get("auth") or {})
    if request.user and getattr(request.user, "is_authenticated", False):
        auth.setdefault("user_id", str(getattr(request.user, "pk", "")) or getattr(request.user, "id", ""))
    return auth


class RelayCaseListAPIView(APIView):
    """GET /api/relay/ — list registered case names and descriptions (no handlers exposed)."""

    def get(self, request: Request) -> Response:
        cases = [
            {"case": c.get("case"), "desc": c.get("desc") or ""}
            for c in get_relay_cases()
        ]
        return Response({"cases": cases})


class RelayCaseAPIView(APIView):
    """
    Generic DRF view for relay cases. Dispatches by case_name (URL kwarg).
    POST body: { "data": {...}, "auth": {...} }. Handler is called with (data=..., auth=...).
    """

    def post(self, request: Request, case_name: str = None) -> Response:
        if not case_name:
            return Response(
                {"detail": "Missing case name in URL."},
                status=status.HTTP_400_BAD_REQUEST,
            )
        try:
            body = dict(request.data) if request.data else {}
        except Exception:
            body = {}
        data = body.get("data")
        if data is None:
            data = {k: v for k, v in body.items() if k != "auth"}
        auth = _build_auth(request, body)
        handler = None
        for c in get_relay_cases():
            if c.get("case") == case_name:
                handler = c.get("func")
                break
        if not handler:
            return Response(
                {"detail": f"Unknown relay case: {case_name}."},
                status=status.HTTP_404_NOT_FOUND,
            )
        try:
            result = handler(data=data, auth=auth)
            if result is None:
                return Response(status=status.HTTP_204_NO_CONTENT)
            if isinstance(result, dict):
                return Response(result, status=status.HTTP_200_OK)
            return Response({"result": result}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"detail": str(e), "case": case_name},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


def build_relay_urlpatterns():
    """Build urlpatterns from all relay cases: list at '' and one POST route per case name."""
    cases = get_relay_cases()
    seen = set()
    patterns = [
        path("", RelayCaseListAPIView.as_view(), name="relay-list"),
    ]
    for c in cases:
        name = c.get("case")
        if not name or name in seen:
            continue
        seen.add(name)
        patterns.append(
            path(f"{name}/", RelayCaseAPIView.as_view(), name=f"relay-{name}"),
        )
    return patterns


urlpatterns = build_relay_urlpatterns()
