# --- Build stage using pixi ---
FROM ghcr.io/prefix-dev/pixi:0.40.0 AS build

WORKDIR /app

# Copy full source and pixi manifest
COPY . /app

# (Optional) run any build tasks defined in pixi.toml
RUN pixi run build

# Install only prod dependencies into a pixi-managed env
RUN pixi install -e prod

# Generate a shell-hook to activate the prod environment
RUN pixi shell-hook -e prod > /shell-hook.sh \
    && echo 'exec "\"$@\""' >> /shell-hook.sh

# --- Production stage without pixi installed ---
FROM python:3.12-slim-bookworm AS production

# Install any minimal OS deps if needed
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the prod environment and activation script from build
COPY --from=build /app/.pixi/envs/prod /app/.pixi/envs/prod
COPY --from=build /shell-hook.sh /shell-hook.sh

# Ensure the pixi env's bin is on PATH
ENV PATH="/app/.pixi/envs/prod/bin:${PATH}"

# Expose the MCP server port (adjust if necessary)
EXPOSE 8080

# Activate the pixi env then run the MCP server module
ENTRYPOINT ["/bin/bash", "/shell-hook.sh"]
CMD ["python", "-m", "dirac_mcp.diracx"]
