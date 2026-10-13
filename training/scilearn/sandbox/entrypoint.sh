#!/bin/bash
# Start the loopback-to-Unix-socket forwarder for the egress proxy, then run the command at nice 19.
set -u
if [ -S /run/hyp/proxy.sock ]; then
  python3 -I /usr/local/lib/scilearn/forwarder.py 127.0.0.1 "${SCILEARN_PROXY_PORT:-3128}" /run/hyp/proxy.sock &
  for _ in $(seq 1 50); do
    if python3 -I -c "import socket,sys; s=socket.socket(); s.settimeout(0.2); s.connect(('127.0.0.1', int(sys.argv[1])))" "${SCILEARN_PROXY_PORT:-3128}" 2>/dev/null; then break; fi
    sleep 0.1
  done
fi
exec nice -n 19 "$@"
