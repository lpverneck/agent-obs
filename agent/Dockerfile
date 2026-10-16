# get Python base image
FROM python:3.13-slim-trixie

# get uv binaries from official uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# set working directory
WORKDIR /agent

# copy dependency files
COPY pyproject.toml uv.lock ./

# install dependencies using uv (using cache strategy to speed up builds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# bootstrap opentelemetry dependencies and add them to the uv environment
# RUN uv run opentelemetry-bootstrap -a requirements | uv add --requirement -

# copy application files
COPY api.py agent.py llm.py tools.py ./

# expose port 8000 for the application
EXPOSE 8002

# set the environment variable for OTEL logging auto-instrumentation
# ENV OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED=true
# ENV OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317
# export metrics every 10s (default is 60s) for faster local feedback
# ENV OTEL_METRIC_EXPORT_INTERVAL=10000

# set entrypoint and default command for the container
# ENTRYPOINT ["uv", "run", "opentelemetry-instrument", "--service_name", "api-service", "fastapi", "run", "--host", "0.0.0.0"]
ENTRYPOINT ["uv", "run", "fastapi", "run", "--host", "0.0.0.0"]
CMD ["--port", "8002"]
