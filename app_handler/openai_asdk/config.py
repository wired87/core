"""
Configuration schema for BestBrain OpenAI Apps SDK deployment.

Ref: https://developers.openai.com/apps-sdk/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AppConfig:
    """App-level configuration for ChatGPT Apps Directory."""

    name: str = "BestBrain"
    version: str = "0.1.0"
    description: str = "Physics simulation and scientific computing environment."
    # Required for submission
    privacy_policy_url: Optional[str] = None
    support_email: Optional[str] = None
    logo_url: Optional[str] = None


@dataclass
class MCPConfig:
    """MCP server configuration."""

    host: str = "0.0.0.0"
    port: int = 8787
    path: str = "/mcp"
    # Stateless mode for simple deployments
    stateless: bool = True


@dataclass
class CSPConfig:
    """Content Security Policy for widget (required for app submission)."""

    connect_domains: List[str] = field(default_factory=lambda: ["https://api.openai.com"])
    resource_domains: List[str] = field(default_factory=list)
    frame_domains: List[str] = field(default_factory=list)
