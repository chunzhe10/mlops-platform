#!/usr/bin/env bash
set -euo pipefail

# Defaults (overridable via environment)
: "${BACKEND_URI:=sqlite:////mlflow/mlflow.db}"
: "${ARTIFACT_ROOT:=file:///mlflow/artifacts}"
: "${PORT:=5000}"
: "${HOST:=0.0.0.0}"

mkdir -p /mlflow/artifacts
# Ensure DB file exists for sqlite
if [[ "$BACKEND_URI" == sqlite:* ]]; then
  # parse path after sqlite:/// or sqlite:////
  # if absolute path like sqlite:////mlflow/mlflow.db then file is /mlflow/mlflow.db
  touch /mlflow/mlflow.db || true
fi

cat <<EOF
Starting MLflow server with:
  backend-store-uri: $BACKEND_URI
  default-artifact-root: $ARTIFACT_ROOT
  host: $HOST
  port: $PORT
EOF

exec mlflow server \
  --backend-store-uri "$BACKEND_URI" \
  --default-artifact-root "$ARTIFACT_ROOT" \
  --host "$HOST" \
  --port "$PORT"
