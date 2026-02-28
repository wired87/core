"""
MCP server for BestBrain as OpenAI Apps SDK app.

Exposes /mcp endpoint per https://developers.openai.com/apps-sdk/quickstart.
Uses FastAPI + uvicorn (project deps). For full MCP spec, add: mcp>=1.0.0

Run: py -m app_handler.openai_asdk.mcp_server --port 8787
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict

# FastAPI/uvicorn are in project r.txt
from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from uvicorn import run as uvicorn_run

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP path required by ChatGPT connector
MCP_PATH = "/mcp"

app = FastAPI(title="BestBrain MCP App", version="0.1.0")


# --- CORS preflight (required for ChatGPT) ---
@app.options(MCP_PATH)
async def mcp_options():
    """CORS preflight for /mcp; required so ChatGPT connector wizard works."""
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, GET, OPTIONS",
            "Access-Control-Allow-Headers": "content-type, mcp-session-id",
            "Access-Control-Expose-Headers": "Mcp-Session-Id",
        },
    )


# --- Health check (GET / returns 200) ---
@app.get("/")
async def root():
    """Health check; ChatGPT may probe this."""
    return PlainTextResponse("BestBrain MCP server")


# --- MCP endpoint stub ---
# Full MCP requires StreamableHTTPServerTransport from mcp package.
# This stub returns a minimal response so the endpoint is reachable.
@app.api_route(MCP_PATH, methods=["GET", "POST", "DELETE"])
async def mcp_handler(request: Request):
    """
    MCP endpoint. ChatGPT connects to https://<your-domain>/mcp.

    To implement full MCP protocol:
    1. pip install mcp
    2. Use mcp.server + StreamableHTTPServerTransport
    3. Handle JSON-RPC 2.0 over HTTP (initialize, tools/list, tools/call)
    """
    # CORS headers for actual requests
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Expose-Headers": "Mcp-Session-Id",
    }

    if request.method == "GET":
        # Some clients send GET for session/health
        return Response(content="OK", status_code=200, headers=headers)

    body = await request.body()
    try:
        data = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return Response(
            content=json.dumps({"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}}),
            status_code=400,
            media_type="application/json",
            headers=headers,
        )

    # Minimal JSON-RPC response; full impl would route to MCP server
    req_id = data.get("id")
    method = data.get("method", "")
    result = {"message": "MCP stub; add mcp package for full protocol", "method": method}
    return Response(
        content=json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}),
        status_code=200,
        media_type="application/json",
        headers=headers,
    )


def main():
    parser = argparse.ArgumentParser(description="BestBrain MCP server for OpenAI Apps SDK")
    parser.add_argument("--port", type=int, default=8787, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    args = parser.parse_args()

    logger.info("BestBrain MCP server listening on http://%s:%s%s", args.host, args.port, MCP_PATH)
    uvicorn_run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
