# get Python base image
FROM python:3.13-slim-trixie

# get uv binaries from official uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# set working directory
WORKDIR /app

# copy dependency files
COPY pyproject.toml uv.lock ./

# install dependencies using uv (using cache strategy to speed up builds)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# copy application files
COPY api.py ./

# expose port 8000 for the application
EXPOSE 8000

# set entrypoint and default command for the container
ENTRYPOINT ["uv", "run", "fastapi", "run", "--host", "0.0.0.0"]
CMD ["--port", "8000"]
