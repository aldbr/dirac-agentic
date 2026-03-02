"""DiracX MCP server entry point.

Supports two transport modes controlled by the MCP_TRANSPORT env var:
- "stdio" (default): Standard I/O transport for CLI and IDE integrations
- "streamable-http": HTTP transport for web deployments

Usage:
    # stdio (default)
    python -m dirac_mcp.server

    # streamable HTTP
    MCP_TRANSPORT=streamable-http MCP_HOST=0.0.0.0 MCP_PORT=8080 python -m dirac_mcp.server

    # Via fastmcp CLI
    fastmcp run dirac-mcp/src/dirac_mcp/server.py
    fastmcp dev dirac-mcp/src/dirac_mcp/server.py
"""

import logging
import os

import dirac_mcp.resources.jobs  # noqa: F401

# Register all tools and resources by importing the modules.
# The decorators in each module register against the shared `mcp` instance.
import dirac_mcp.tools.jobs  # noqa: F401

# Import the shared FastMCP instance
from dirac_mcp.app import mcp  # noqa: F401

logger = logging.getLogger(__name__)


def _check_credentials() -> None:
    """Log warnings if DiracX credential env vars are missing."""
    if not os.environ.get("DIRACX_URL"):
        logger.warning(
            "DIRACX_URL is not set — tools requiring DiracX access will fail. "
            "Set it to your DiracX instance URL."
        )
    if not os.environ.get("DIRACX_CREDENTIALS_PATH"):
        logger.warning(
            "DIRACX_CREDENTIALS_PATH is not set — tools requiring DiracX access will fail. "
            "Set it to the path of your credentials file."
        )


def main() -> None:
    """Run the MCP server with configurable transport."""
    _check_credentials()
    transport = os.environ.get("MCP_TRANSPORT", "stdio")

    if transport == "streamable-http":
        host = os.environ.get("MCP_HOST", "0.0.0.0")
        port = int(os.environ.get("MCP_PORT", "8080"))
        mcp.run(transport="streamable-http", host=host, port=port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
