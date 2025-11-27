ML Ops Platform — Local Stack for Experiment Tracking and Inference

Overview
- A minimal, batteries-included local MLOps stack to track experiments with MLflow and serve models with NVIDIA Triton Inference Server.
- Designed for fast onboarding: one command to bring up the stack, clear separation per component, and CI validation for Docker Compose config.

Core Components
- MLflow: experiment tracking UI and REST API.
- Triton Inference Server: production-grade model serving (HTTP/gRPC + Prometheus metrics).

Tech Stack
- Docker + Docker Compose (profiles; root `include` for component files)
- GitHub Actions (validate Compose configuration on push/PR)

Repository Structure
```
.
├─ docker-compose.yml              # Root compose that includes component compose files
├─ .github/workflows/validate-compose.yml
├─ mlflow/
│  ├─ Dockerfile
│  ├─ entrypoint.sh
│  ├─ docker-compose.yml          # Profiles: ["full", "mlflow"]
│  ├─ PROFILES.md                 # Usage with profiles and root include (see link below)
│  └─ data/                       # Local DB + artifacts (mapped to /mlflow)
└─ triton/
	 ├─ Dockerfile
	 ├─ docker-compose.yml          # Profiles: ["full", "triton"]
	├─ PROFILES.md                 # Usage with profiles and root include (see link below)
	 ├─ add_dummy_model.sh          # Helper to download/add a sample model
	 └─ models/                     # Model repository (ignored in Git; populated by scripts)
```

Compose Profiles and Root Include
- Each component defines its own compose file with profiles for flexible startup.
- The root `docker-compose.yml` uses Compose `include` (Compose v2.20+) to reference component files, so you don’t need multiple `-f` flags.

Quick Start (from repo root)
```bash
# Start the full stack (MLflow + Triton)
docker compose --profile full up -d --build

# Start only MLflow
docker compose --profile mlflow up -d --build

# Start only Triton
docker compose --profile triton up -d --build

# Validate the configuration (merges includes and profiles)
docker compose config
```

MLflow Details
- UI: `http://localhost:5000`
- Config is controlled via environment variables in the compose file (e.g., `BACKEND_URI`, `ARTIFACT_ROOT`).
- Persistent data mapped under `mlflow/data`.
 - Onboarding: see `mlflow/PROFILES.md` for profile-based commands and tips.

Triton Details
- Endpoints: HTTP 8000, gRPC 8001, Metrics 8002.
- Place model repositories under `triton/models/<model_name>/` or use `triton/add_dummy_model.sh` to fetch a sample.
- Large model files are ignored via `.gitignore`; prefer script-based download or Git LFS if you must track them.
 - Onboarding: see `triton/PROFILES.md` for profile-based commands and tips.

CI: Compose Validation
- GitHub Actions workflow validates compose configuration on push/PR:
	- Validates the root compose (with `include`).
	- Discovers all component `docker-compose.yml` files and runs `docker compose config` against each.
	- Ensures early detection of syntax or merge issues.

Conventions
- Component-local compose files live next to their Dockerfiles.
- Use profiles for conditional startup: `full`, `mlflow`, `triton`.
- Avoid committing large models; use scripts or LFS.

Next Steps
- Add more components (e.g., MinIO/S3 for MLflow artifacts, Postgres for MLflow backend).
- Extend CI to lint Dockerfiles or run health checks against services.
