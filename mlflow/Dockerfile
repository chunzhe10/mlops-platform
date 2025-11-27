FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# Install mlflow and a small web server if needed
RUN pip install --no-cache-dir mlflow==3.6.0

WORKDIR /app

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/mlflow"]
EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
