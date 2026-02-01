# Use an official Python runtime as a parent image
FROM python:3.14-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (if any specific ones are needed for sas7bdat or others)
# RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

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
