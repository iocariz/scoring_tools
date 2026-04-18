# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies.
# - libgomp1 is required by lightgbm (OpenMP runtime); python:3.12-slim does
#   not include it, and the CI Docker test job was failing with
#   `OSError: libgomp.so.1: cannot open shared object file` prior to this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set work directory
WORKDIR /app

# Copy dependency definition
COPY pyproject.toml .

# Install dependencies
# We use --system to install into the system python, avoiding venv complexity in Docker
# and --no-dev to keep the image smaller if dev deps aren't needed for prod
RUN uv pip install --system -e .

# Copy project files
COPY . .

# Create directories for mounting
RUN mkdir -p data output

# Set default command (can be overridden)
CMD ["python", "main.py"]
