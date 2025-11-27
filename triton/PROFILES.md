Profile-in-component: Start full stack (mlflow + triton)

This repository uses the profile-in-component approach with an `include` directive at the root for simplified commands.

Commands (from repo root)

```bash
# Start only triton
docker compose --profile triton up -d --build

# Start only mlflow
docker compose --profile mlflow up -d --build

# Start the full stack (both services)
docker compose --profile full up -d --build

# Validate the configuration
docker compose config
```

Commands (from triton/ directory)

```bash
# Start only triton (local compose file)
docker compose --profile triton up -d --build
```

Notes

- No duplication — each service is defined once in its component file.
- The root `docker-compose.yml` uses `include` to reference component files (requires Compose v2.20+).
- Services won't start without specifying a profile (e.g., `--profile full`).
