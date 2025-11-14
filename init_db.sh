#!/bin/bash
set -eo pipefail

echo "init.sh: waiting for PostgreSQL unix socket and readiness (timeout 120s)..."

# wait for unix socket file to exist
timeout=120
count=0
sock="/var/run/postgresql/.s.PGSQL.5432"

while [ $count -lt $timeout ]; do
  if [ -S "$sock" ]; then
    echo "init.sh: socket found: $sock"
    break
  fi
  sleep 1
  count=$((count+1))
done

if [ $count -ge $timeout ]; then
  echo "init.sh: timeout waiting for unix socket ($sock). Exiting with failure." >&2
  exit 1
fi

# Now use pg_isready with default socket (no -h). Retry a few times.
count=0
while [ $count -lt $timeout ]; do
  if pg_isready -U "${POSTGRES_USER:-postgres}" >/dev/null 2>&1; then
    echo "init.sh: pg_isready reports Postgres ready"
    break
  fi
  sleep 1
  count=$((count+1))
done

if [ $count -ge $timeout ]; then
  echo "init.sh: timeout waiting for pg_isready to report ready. Exiting." >&2
  exit 1
fi

echo "init.sh: Postgres ready. Running seed script with venv"
/app/venv/bin/python /app/seed_db.py
echo "init.sh: seed script finished."
