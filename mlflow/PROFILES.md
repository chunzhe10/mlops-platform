Profile-in-component: Start full stack (mlflow + triton)

This repository uses the profile-in-component approach with an `include` directive at the root for simplified commands.

Examples (from repo root)

```bash
# Start only mlflow
docker compose --profile mlflow up -d --build

# Start the full stack
docker compose --profile full up -d --build

# Validate configuration
docker compose config
```
