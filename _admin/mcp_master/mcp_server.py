#!/usr/bin/env python3
"""
Minimal MCP stdio server for QBrain relay cases.

Run: py mcp_server.py (or python mcp_server.py)
Expects MCP protocol over stdin/stdout per Model Context Protocol spec.
"""
import json
import sys


def main():
    """Placeholder: read/write MCP messages on stdio."""
    for line in sys.stdin:
        try:
            msg = json.loads(line.strip())
            # Echo back or handle MCP initialize/list_tools
            if msg.get("method") == "initialize":
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0",
                    "id": msg.get("id"),
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "serverInfo": {"name": "qbrain-relay", "version": "1.0.0"},
                    },
                }) + "\n")
            elif msg.get("method") == "tools/list":
                sys.stdout.write(json.dumps({
                    "jsonrpc": "2.0",
                    "id": msg.get("id"),
                    "result": {"tools": []},
                }) + "\n")
            sys.stdout.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
