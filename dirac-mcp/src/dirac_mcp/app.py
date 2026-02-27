"""FastMCP application instance.

All tool, prompt, and resource modules import `mcp` from here
to register their decorators against a single shared instance.
"""

from fastmcp import FastMCP

mcp = FastMCP("DiracX Services")
