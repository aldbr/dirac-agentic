# --- Build stage using pixi ---
FROM ghcr.io/prefix-dev/pixi:0.40.0 AS build

WORKDIR /app

# Copy the project root's pixi.toml and pyproject.toml
COPY ./pixi.toml /app/
COPY ./pyproject.toml /app/

# Copy full source and pixi manifest
COPY ./dirac-mcp /app/dirac-mcp
COPY ./dirac-dataset /app/dirac-dataset

# Install only dirac-mcp dependencies into a pixi-managed env
RUN pixi install -e dirac-mcp

# --- Production stage without pixi installed ---
FROM python:3.12-slim-bookworm AS production

# 1) Install minimal OS deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copy over the pixi-built env (dirac-mcp) and drop the shell hook entirely
COPY --from=build /app/.pixi/envs/dirac-mcp /app/.pixi/envs/dirac-mcp
COPY --from=build /app/dirac-mcp      /app/dirac-mcp

# 3) Put that env's bin on PATH
ENV PATH="/app/.pixi/envs/dirac-mcp/bin:${PATH}"

# 4) Configure for streamable HTTP transport
ENV MCP_TRANSPORT=streamable-http
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8080

# 5) Expose your MCP port
EXPOSE 8080

# 6) Directly invoke python from the Pixi env
ENTRYPOINT ["python", "-m", "dirac_mcp.server"]
