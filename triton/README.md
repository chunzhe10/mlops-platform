Triton inference server — local dockerized scaffold

This folder contains a minimal scaffold to fetch the Triton Inference Server repository and build/run a containerized Triton server.

What I added

- `Dockerfile` — simple image that layers on an official Triton runtime image and copies `models/` into the container.
- `docker-compose.yml` — example compose file to build/run the image and expose Triton ports.
- `models/` — directory (create model repositories here). Place model repositories under `triton/models/<your-model-repo>`.

Quick start

1) Prepare a model repository

Put one or more model repositories into `triton/models/`. Each model repo should follow Triton's model repository layout. Example:

```
triton/models/my_model/1/model.plan
triton/models/my_model/config.pbtxt
```

2) Build and run the container (uses `BASE_IMAGE` arg — default points at an NVIDIA runtime image):

From the `triton/` directory:

```bash
# Optionally override the base image via env var (use a tag you can access)
export TRITON_BASE_IMAGE=nvcr.io/nvidia/tritonserver:23.05-py3

docker compose up --build
```

3) Access Triton

- HTTP inference endpoint: `http://localhost:8000/v2/models/<model>/infer`
- gRPC: `localhost:8001`
- Metrics: `http://localhost:8002/metrics`

Notes and next steps

- The default `Dockerfile` uses an NVIDIA Triton runtime image (NGC). Pulling that image may require access to the NVIDIA container registry or switching to a public image if available.
- For development, mounting `triton/models` as a volume is recommended instead of baking models into the image.
- I can add an example `models/` package (a small dummy model), or GPU/CPU-specific compose profiles. Which would you like next?
