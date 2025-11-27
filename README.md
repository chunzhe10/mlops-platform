Project README — Profile-in-component approach

This repository uses component-local Compose files with profiles and an `include` directive at the root for simplified commands:

- `mlflow/docker-compose.yml` — defines `mlflow` service with profiles: `["full", "mlflow"]`
- `triton/docker-compose.yml` — defines `triton` service with profiles: `["full", "triton"]`
- `docker-compose.yml` (root) — includes both component files

Usage examples

```bash
# Start the full stack (from repo root)
docker compose --profile full up -d --build

# Start only mlflow
docker compose --profile mlflow up -d --build

# Start only triton
docker compose --profile triton up -d --build

# Validate the configuration
docker compose config
```

Notes

- No duplication — each service is defined once in its component file.
- The root `docker-compose.yml` uses the `include` directive (requires Compose v2.20+).
- Services with profiles won't start without specifying a profile.
