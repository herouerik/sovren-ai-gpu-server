#!/usr/bin/env bash
# Stops every process bound to this app's port, however it was started.
#
# `uvicorn --reload` runs two processes -- a reloader and a separate worker
# -- that share the same listening socket fd (the worker inherits it from
# the reloader). If only the reloader dies (e.g. a Ctrl-C that doesn't
# reach the worker, or the terminal it was running in just closing), the
# worker survives on its own and keeps holding the port -- the next
# `./run.sh` then fails to bind, or worse, silently starts a second
# instance while a stale one keeps answering requests with pre-update
# code. This has been hit repeatedly by hand this session; always kill
# every pid actually holding the port, not just whichever pid you happen
# to know about.
#
# Safe to run any time, including when nothing is listening.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PORT="${SOVREN_PORT:-8082}"

pids_on_port() {
    ss -ltnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | sort -u
}

pids=$(pids_on_port)
if [ -z "$pids" ]; then
    echo "Nothing listening on port ${PORT}."
    exit 0
fi

echo "Stopping process(es) on port ${PORT}: $(echo "$pids" | tr '\n' ' ')"
kill -TERM $pids 2>/dev/null

for _ in $(seq 1 10); do
    [ -z "$(pids_on_port)" ] && break
    sleep 0.3
done

remaining=$(pids_on_port)
if [ -n "$remaining" ]; then
    echo "Still alive after SIGTERM, force killing: $(echo "$remaining" | tr '\n' ' ')"
    kill -KILL $remaining 2>/dev/null
    sleep 0.3
fi

if [ -n "$(pids_on_port)" ]; then
    echo "WARNING: something is still listening on port ${PORT}." >&2
    exit 1
fi
echo "Port ${PORT} is free."
