"""Allowlisting HTTP CONNECT proxy on a Unix socket, for proposer sandboxes.

The sandbox has no network. Its only egress is this proxy, reached through a
bind-mounted Unix socket. Rules:
  - only CONNECT, only port 443;
  - the hostname must match ALLOW_EXACT or end with one of ALLOW_SUFFIX, and not be
    in DENY_EXACT (the claude.ai MCP connector proxy);
  - the name is resolved here, and the connection goes to a resolved address
    only if every resolved address is globally routable (no loopback, RFC1918,
    link-local, CGNAT 100.64/10 incl. Tailscale, IPv6 ULA, multicast, reserved);
  - IP literals are refused.
Every request is logged as one JSON line (allowed or denied).

Usage: python3 proxy.py --socket <session>/proxy.sock --log <session>/logs/proxy.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import ipaddress
import json
import os
import signal
import socket
import time
from typing import Callable, Iterable

ALLOW_EXACT = {"api.anthropic.com", "claude.ai", "claude.com"}
ALLOW_SUFFIX = (".anthropic.com", ".claude.ai", ".claude.com")
ALLOW_PORTS = {443}
# claude.ai MCP connectors (Drive, Docs, ...) would be a side channel around the broker.
DENY_EXACT = {"mcp-proxy.anthropic.com"}

CGNAT = ipaddress.ip_network("100.64.0.0/10")
ULA = ipaddress.ip_network("fc00::/7")


def host_allowed(host: str) -> bool:
    h = host.lower().rstrip(".")
    if not h or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-." for c in h):
        return False
    if h in DENY_EXACT:
        return False
    return h in ALLOW_EXACT or any(h.endswith(s) for s in ALLOW_SUFFIX)


def ip_public(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr.split("%")[0])
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast:
        return False
    if ip.is_reserved or ip.is_unspecified:
        return False
    if ip.version == 4 and ip in CGNAT:
        return False
    if ip.version == 6 and ip in ULA:
        return False
    return ip.is_global


def default_resolve(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return sorted({i[4][0] for i in infos})


def check_target(host: str, port: int,
                 resolve: Callable[[str, int], Iterable[str]] = default_resolve):
    """Return (ok, reason, addresses). Pure policy; used by the server and the tests."""
    try:
        ipaddress.ip_address(host.strip("[]"))
        return False, "ip_literal", []
    except ValueError:
        pass
    if port not in ALLOW_PORTS:
        return False, "port", []
    if not host_allowed(host):
        return False, "host_not_allowlisted", []
    try:
        addrs = list(resolve(host, port))
    except OSError as e:
        return False, f"resolve_failed:{e.__class__.__name__}", []
    if not addrs:
        return False, "no_address", []
    bad = [a for a in addrs if not ip_public(a)]
    if bad:
        return False, "non_public_address", addrs
    return True, "ok", addrs


class Proxy:
    def __init__(self, log_path: str, resolve=default_resolve):
        self.log_path = log_path
        self.resolve = resolve
        self.log_fh = open(log_path, "a", buffering=1)

    def log(self, **rec):
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.log_fh.write(json.dumps(rec) + "\n")

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=30)
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ConnectionError):
            writer.close()
            return
        line = head.split(b"\r\n", 1)[0].decode("latin1", "replace")
        parts = line.split()
        if len(parts) != 3 or parts[0].upper() != "CONNECT":
            self.log(decision="deny", reason="method", request=line[:200])
            await self._reply(writer, 405, "Method Not Allowed")
            return
        target = parts[1]
        host, _, port_s = target.rpartition(":")
        try:
            port = int(port_s)
        except ValueError:
            self.log(decision="deny", reason="bad_target", target=target[:200])
            await self._reply(writer, 400, "Bad Request")
            return
        loop = asyncio.get_running_loop()
        ok, reason, addrs = await loop.run_in_executor(None, check_target, host, port, self.resolve)
        if not ok:
            self.log(decision="deny", reason=reason, host=host[:200], port=port, addrs=addrs)
            await self._reply(writer, 403, "Forbidden")
            return
        upstream = None
        last_err = None
        for a in addrs:
            try:
                upstream = await asyncio.wait_for(asyncio.open_connection(a, port), timeout=15)
                break
            except (OSError, asyncio.TimeoutError) as e:
                last_err = e
        if upstream is None:
            self.log(decision="error", reason=f"connect_failed:{last_err.__class__.__name__}", host=host, port=port)
            await self._reply(writer, 502, "Bad Gateway")
            return
        self.log(decision="allow", host=host, port=port, addr=upstream[1].get_extra_info("peername")[0])
        writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        await writer.drain()
        ur, uw = upstream
        await asyncio.gather(self._pipe(reader, uw), self._pipe(ur, writer))

    @staticmethod
    async def _reply(writer, code, text):
        try:
            writer.write(f"HTTP/1.1 {code} {text}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".encode())
            await writer.drain()
        except (ConnectionError, OSError):
            pass
        writer.close()

    @staticmethod
    async def _pipe(r, w):
        try:
            while True:
                data = await r.read(65536)
                if not data:
                    break
                w.write(data)
                await w.drain()
        except (ConnectionError, OSError):
            pass
        finally:
            try:
                w.close()
            except Exception:
                pass


async def serve(sock_path: str, log_path: str):
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    proxy = Proxy(log_path)
    server = await asyncio.start_unix_server(proxy.handle, path=sock_path)
    os.chmod(sock_path, 0o600)
    proxy.log(decision="start", socket=sock_path, pid=os.getpid())
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop.set)
    async with server:
        await stop.wait()
    proxy.log(decision="stop")
    try:
        os.unlink(sock_path)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--socket", required=True)
    ap.add_argument("--log", required=True)
    a = ap.parse_args()
    asyncio.run(serve(a.socket, a.log))


if __name__ == "__main__":
    main()
