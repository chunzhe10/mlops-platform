# Simple Triton image builder that layers on the official Triton runtime
# Default base image is an NVIDIA Triton runtime image. Replace with a tag you have access to.

ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:23.05-py3
FROM ${BASE_IMAGE}

# Author: scaffold added by repo
LABEL maintainer="mlops-platform"

# Copy local models into the image (optional). Prefer mounting a model repo via volume in production.
COPY models /models

# Default model repository path and ports
ENV TRITON_MODEL_REPOSITORY=/models

EXPOSE 8000 8001 8002

ENTRYPOINT ["/opt/tritonserver/bin/tritonserver", "--model-repository=/models"]
