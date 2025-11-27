MLflow Dockerized setup

This folder contains files to build and run a simple Dockerized MLflow tracking server using a local SQLite backend and a local artifact store.

Files added

- `Dockerfile` — builds an image with MLflow installed.
- `entrypoint.sh` — starts the `mlflow server` using environment variables.
- `../docker-compose.yml` — Compose file (at repo root) that builds and runs the container.
- `../.env` — environment variables used by `docker-compose`.

Data persistence

Data is persisted under `mlflow/data` (mapped to `/mlflow` inside the container). It contains the SQLite DB file and the `artifacts/` directory for uploaded artifacts.

Quick start

1) Build and start with Docker Compose (runs in foreground):

```bash
docker compose up --build
```

2) Run detached:

```bash
docker compose up -d --build
```

3) Access MLflow UI in your browser at: `http://localhost:5000`

4) Stopping and removing containers:

```bash
docker compose down
```

Changing backend/storage

- To use PostgreSQL or MySQL, change `BACKEND_URI` in `.env` to a supported SQLAlchemy URI and add the corresponding DB service to `docker-compose.yml`.
- To use S3 for artifacts, set `ARTIFACT_ROOT` to an S3 URI (e.g. `s3://my-bucket/mlflow`) and provide credentials via environment variables or credentials file.

Notes

- The Dockerfile pins MLflow to `2.6.2`; change as needed.
- For production use, protect MLflow behind authentication (proxy or OAuth), use a proper DB, and secure artifact storage.
