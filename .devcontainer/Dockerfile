# Dev container for RAG-PROJECT
FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.11

# Avoid running pip as root in the image build to keep it simple; we'll install requirements in postCreateCommand
# Copy requirements so that developers can optionally build an image with deps installed
COPY requirements.txt /tmp/requirements.txt

# Install common tooling (optional) and pre-install minimal packages to speed up postCreateCommand
RUN apt-get update \
  && apt-get install -y --no-install-recommends git build-essential libgl1 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Default to a long-running sleep to keep container alive if started directly
CMD ["sleep", "infinity"]
