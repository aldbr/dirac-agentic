# dirac-mcp: Standalone MCP Server & Future DiracX Extension (status: **accepted**)

|              |                                                                                          |
|--------------|------------------------------------------------------------------------------------------|
| **Date**     | 2026-02-27                                                                                |
| **Status**   | **Accepted**                                                                              |
| **Stake**    | Define how dirac-mcp operates today and how it will integrate into DiracX                 |
| **Context**  | We need an MCP server exposing DiracX functionality to AI agents. It must work standalone for development and testing, and eventually integrate into DiracX deployments without code duplication. |

---

## 1 Problem

AI agents need a standard interface to interact with DiracX (job management, monitoring, etc.). The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) provides that standard. The question is where this server lives and how it authenticates:

- **Standalone**: Easy to develop and test, but requires separate credential management.
- **Inside DiracX**: Inherits authentication and deployment, but couples release cycles.
- **Extension package**: Best of both worlds — same code, two deployment modes.

## 2 Decision

**dirac-mcp is a standalone Python package that can also be mounted into DiracX as an extension.**

### Standalone mode (current)

```
┌──────────────┐     stdio / HTTP      ┌──────────────┐
│   AI Agent   │ ◄──────────────────►  │  dirac-mcp   │
│ (Claude, etc)│      MCP protocol     │  (FastMCP)   │
└──────────────┘                       └──────┬───────┘
                                              │ HTTPS
                                              ▼
                                       ┌──────────────┐
                                       │    DiracX     │
                                       │    API        │
                                       └──────────────┘
```

- Runs via `python -m dirac_mcp.server` or `fastmcp run`
- Transports: `stdio` (IDE integration) or `streamable-http` (web deployment)
- Authentication: reads `DIRACX_URL` and `DIRACX_CREDENTIALS_PATH` env vars, uses `AsyncDiracClient()` from the `diracx` package

### DiracX extension mode (future)

```
┌──────────────┐     stdio / HTTP      ┌───────────────────────────────────┐
│   AI Agent   │ ◄──────────────────►  │           DiracX                  │
│ (Claude, etc)│      MCP protocol     │  ┌─────────────────────────────┐  │
└──────────────┘                       │  │  /api/mcp  (dirac-mcp)     │  │
                                       │  │  FastAPI router via         │  │
                                       │  │  entrypoints               │  │
                                       │  └─────────────────────────────┘  │
                                       │  JWT auth middleware applied      │
                                       └───────────────────────────────────┘
```

DiracX uses Python entrypoints to discover extension packages. When `dirac-mcp` is installed in a DiracX deployment, it registers a FastAPI router:

```toml
# In dirac-mcp/pyproject.toml
[project.entry-points."diracx.extensions"]
mcp = "dirac_mcp.extension:router"
```

The extension module would expose the FastMCP app as a FastAPI-compatible ASGI app via `mcp.http_app()`, and DiracX's `create_app()` would mount it with JWT authentication middleware automatically applied.

**Key property**: the MCP protocol layer (tool schemas, prompts, resources) is shared between both modes. What changes is how DiracX is called.

## 3 Architecture

### What dirac-mcp actually provides

The value of dirac-mcp is the **MCP protocol adapter**, not the business logic:

- **Tool schemas**: Parameter types, descriptions, return formats that AI agents understand
- **Prompt definitions**: Guides that help agents formulate searches, analyze failures, create JDL
- **Resource URIs**: `dirac-job://{id}`, `dirac-dashboard://jobs`
- **JDL generation**: `create_basic_jdl()` is pure logic, useful in both modes

The tool implementations themselves are thin — they translate MCP calls into DiracX operations.

### Standalone mode (current)

Tools call DiracX over HTTP via `AsyncDiracClient()`:

```python
# tools/jobs.py — standalone mode
@mcp.tool()
async def search_jobs(conditions, ...):
    async with AsyncDiracClient() as client:          # HTTP to remote DiracX
        jobs, _ = await client.jobs.search(...)
```

### Extension mode (future)

When running inside DiracX, there's no need for HTTP round-trips. Tools would call DiracX's internal service layer directly:

```python
# tools/jobs.py — extension mode
@mcp.tool()
async def search_jobs(conditions, ...):
    jobs = await job_service.search(...)               # Direct Python call
```

This is the main architectural benefit of the extension approach: **no HTTP overhead, no serialization, direct access to DiracX's service layer** with the authenticated user context provided by FastAPI dependency injection.

### Current structure

```
dirac-mcp/src/dirac_mcp/
├── app.py              # Shared FastMCP instance
├── server.py           # Standalone entry point (stdio + streamable HTTP)
├── tools/
│   └── jobs.py         # Tool implementations (currently uses AsyncDiracClient)
├── prompts/
│   └── jobs.py         # Prompt definitions (shared by both modes)
└── resources/
    └── jobs.py         # Resource definitions (shared by both modes)
```

All modules register against the shared `mcp` instance in `app.py` via decorators. The entry point (`server.py`) imports these modules to trigger registration, then starts the appropriate transport.

### Migration path to extension mode

When implementing the extension, the tool layer will need refactoring:

1. **Prompts and resources**: No change needed — they're already mode-agnostic
2. **`create_basic_jdl`**: No change needed — pure function, no DiracX calls
3. **`search_jobs`, `get_job`, `submit_job`, `get_job_status_summary`**: Replace `AsyncDiracClient()` HTTP calls with direct imports from DiracX's service layer

The refactoring should be straightforward since the tools are thin wrappers. The complexity is in understanding DiracX's internal service API, not in the MCP layer itself.

## 4 Authentication

### Standalone mode

Uses the `diracx` client library which reads credentials from environment:

| Variable | Purpose |
|----------|---------|
| `DIRACX_URL` | DiracX instance URL |
| `DIRACX_CREDENTIALS_PATH` | Path to credentials JSON file |

The user must authenticate with DiracX (`diracx login`) before starting the MCP server. The MCP server itself has no auth layer — it runs as a single-identity process with pre-obtained credentials.

### DiracX extension mode (future)

No credential env vars needed. DiracX's JWT middleware authenticates every request, and the authenticated user context is passed to tool functions via the FastAPI dependency injection system.

## 5 Why not embed directly in DiracX?

- **Release cadence**: dirac-mcp can iterate faster than DiracX core.
- **Optional dependency**: Not all DiracX deployments need MCP.
- **Development experience**: Standalone mode allows testing without a full DiracX stack.
- **Framework independence**: MCP tools can be tested and used with any MCP client, not just through DiracX HTTP.

## 6 Next steps

1. ~~Build standalone mode with FastMCP 3.0~~ (done)
2. Test with real DiracX instance
3. Study DiracX's internal service API (`diracx.routers.jobs`, etc.) to understand the direct call surface
4. Refactor tools to call DiracX service layer directly (replacing `AsyncDiracClient`)
5. Implement `extension.py` module with FastAPI router
6. Add `diracx.extensions` entrypoint to pyproject.toml
7. Test dual-mode deployment
