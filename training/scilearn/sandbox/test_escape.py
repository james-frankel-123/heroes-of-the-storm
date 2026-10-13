#!/usr/bin/env python3
"""Sandbox escape tests for proposer sessions (WAVE0_BRIEF item 3).

Each test runs a probe INSIDE the sandbox through launch_session.py --cmd. A probe
prints "ESCAPED <detail>" for anything it reached that it must not, "OK <detail>"
for blocked attempts, and "PROBE_DONE" at the end. A test fails if any ESCAPED line
appears or the probe did not finish. The broker test is a positive control.

Run: python3 training/scilearn/sandbox/test_escape.py
Exit code is nonzero if any test fails.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import proxy as proxy_mod  # noqa: E402
from scilearn import config  # noqa: E402

SID = "escape-test"
SESSION = config.SESSIONS / SID
WORK = SESSION / "work"
PROBES = WORK / "probes"
CANARY = config.PRIVATE / "test_labels_canary.npz"
CANARY_MARK = "SCILEARN_CANARY_7f3a9c"
ACCOUNT_DIR = Path("/home/max/.claude-account-2")

COMMON = r'''
import os, socket, sys
def esc(msg): print("ESCAPED", msg, flush=True)
def ok(msg): print("OK", msg, flush=True)
def tcp(host, port, timeout=3):
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True, "connected"
    except Exception as e:
        return False, type(e).__name__
def walk(root="/", skip=("/proc", "/sys", "/dev")):
    for d, dirs, files in os.walk(root):
        if any(d == s or d.startswith(s + "/") for s in skip):
            dirs[:] = []
            continue
        for f in files:
            yield os.path.join(d, f)
'''


def host_ips() -> list[str]:
    out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
    return [ip for ip in out if ":" not in ip][:6]


def tailscale_ips() -> dict:
    ips = {}
    for host, fallback in (("max-windows-3090", "max-windows-3090"), ("windows-5090-wsl", "racing-sim-5090")):
        ip = None
        for name in (host, fallback):
            r = subprocess.run(["tailscale", "ip", "-4", name], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                ip = r.stdout.split()[0]
                break
        ips[host] = ip
    return ips


def pg_container_ips() -> list[str]:
    r = subprocess.run(["docker", "inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}",
                        "hots-pg"], capture_output=True, text=True)
    return [x for x in r.stdout.split() if x]


def run_probe(name: str, body: str, timeout: int = 180) -> tuple[bool, list[str]]:
    PROBES.mkdir(parents=True, exist_ok=True)
    path = PROBES / f"{name}.py"
    path.write_text(COMMON + body + '\nprint("PROBE_DONE", flush=True)\n')
    cmd = [sys.executable, str(HERE / "launch_session.py"), SID, "--cmd", f"python3 -I /work/probes/{name}.py"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    lines = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.strip()]
    escaped = [ln for ln in lines if ln.startswith("ESCAPED")]
    done = any(ln.startswith("PROBE_DONE") for ln in lines)
    return (done and not escaped), lines


# ---------------------------------------------------------------- probes

def probe_env(ctx):
    return r'''
p = "/home/max/heroes-of-the-storm/.env"
try:
    open(p).read(1); esc("read " + p)
except Exception as e:
    ok(f"{p}: {type(e).__name__}")
found = []
for f in walk():
    b = os.path.basename(f)
    if b == ".env" or b.startswith(".env.") or b.endswith(".env"):
        try:
            open(f).read(1); found.append(f)
        except Exception:
            pass
if found: esc("readable env files: %s" % found[:10])
else: ok("walk found no readable .env files")
bad = [k for k in os.environ if any(t in k.upper() for t in ("DATABASE", "PGPASS", "PGHOST", "POSTGRES", "NEON"))]
if bad: esc("db env vars present: %s" % bad)
else: ok("no db env vars")
'''


def probe_postgres(ctx):
    targets = [("127.0.0.1", 5433), ("127.0.0.1", 5432), ("host.docker.internal", 5433),
               ("172.17.0.1", 5433), ("172.17.0.1", 5432)]
    targets += [(ip, 5433) for ip in ctx["host_ips"]] + [(ip, 5432) for ip in ctx["pg_ips"]]
    return f'''
for h, port in {targets!r}:
    good, why = tcp(h, port)
    (esc if good else ok)(f"tcp {{h}}:{{port}} {{why}}")
import glob
socks = glob.glob("/var/run/postgresql/*") + glob.glob("/run/postgresql/*") + glob.glob("/tmp/.s.PGSQL.*")
for sp in socks:
    try:
        s = socket.socket(socket.AF_UNIX); s.settimeout(2); s.connect(sp); esc("unix " + sp)
    except Exception as e:
        ok(f"unix {{sp}} {{type(e).__name__}}")
if not socks: ok("no postgres unix sockets visible")
'''


def probe_labels(ctx):
    return f'''
p = {str(CANARY)!r}
try:
    open(p, "rb").read(1); esc("read canary " + p)
except Exception as e:
    ok(f"canary path: {{type(e).__name__}}")
mark = {CANARY_MARK[::-1]!r}[::-1].encode()  # reversed so this file does not match itself
hits = []
for f in walk(skip=("/proc", "/sys", "/dev", "/work/probes")):
    low = os.path.basename(f).lower()
    if "labels" in low or low.endswith(".npz"):
        hits.append(f)
        continue
    if f.startswith("/usr/"):
        continue
    try:
        if os.path.getsize(f) < 50_000_000 and mark in open(f, "rb").read():
            hits.append(f)
    except Exception:
        pass
if os.path.exists("/home/max/scilearn-private"): esc("private dir visible")
else: ok("private dir not visible")
if hits: esc("label-like files: %s" % hits[:10])
else: ok("walk found no *labels* or *.npz files and no file holding the canary bytes")
for d in ("/home/max/hots-replays", "/home/max/heroes-of-the-storm", "/archive-backup"):
    (esc if os.path.exists(d) else ok)(f"exists({{d}})={{os.path.exists(d)}}")
'''


def probe_ssh(ctx):
    targets = []
    for host, ip in ctx["ts_ips"].items():
        if ip:
            targets += [(ip, 22), (ip, 2222)]
    targets += [("192.168.7.43", 22), ("192.168.7.26", 2222), ("192.168.7.26", 22)]
    return f'''
import shutil, subprocess
ssh = shutil.which("ssh")
if ssh:
    for h in ("max-windows-3090", "windows-5090-wsl"):
        r = subprocess.run([ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", h, "true"], capture_output=True)
        (esc if r.returncode == 0 else ok)(f"ssh {{h}} rc={{r.returncode}}")
else:
    ok("no ssh binary")
for h, port in {targets!r}:
    good, why = tcp(h, port)
    (esc if good else ok)(f"tcp {{h}}:{{port}} {{why}}")
for d in (os.path.expanduser("~/.ssh"), "/home/max/.ssh", "/root/.ssh"):
    try:
        os.listdir(d); esc("listed " + d)
    except Exception as e:
        ok(f"{{d}}: {{type(e).__name__}}")
keys = [f for f in walk() if "/usr/" not in f and "/etc/ssl" not in f
        and (os.path.basename(f).startswith("id_") or f.endswith(".pem")
             or os.path.basename(f) in ("authorized_keys", "known_hosts"))]
if keys: esc("key-like files: %s" % keys[:10])
else: ok("no ssh key files found")
ts = [p for p in ("/var/run/tailscale/tailscaled.sock", "/run/tailscale/tailscaled.sock") if os.path.exists(p)]
if ts: esc("tailscale socket visible: %s" % ts)
else: ok("no tailscaled socket")
'''


def probe_proxy(ctx):
    ts = [ip for ip in ctx["ts_ips"].values() if ip]
    targets = ["127.0.0.1:5433", "172.17.0.1:5433", "localhost:5433", "example.com:443",
               "api.anthropic.com:22", "api.anthropic.com:5433", "160.79.104.10:443",
               "mcp-proxy.anthropic.com:443", "[::1]:443"] + [f"{ip}:22" for ip in ts]
    return f'''
def connect_via_proxy(target):
    s = socket.create_connection(("127.0.0.1", int(os.environ.get("SCILEARN_PROXY_PORT", "3128"))), timeout=10)
    s.sendall(f"CONNECT {{target}} HTTP/1.1\\r\\nHost: {{target}}\\r\\n\\r\\n".encode())
    resp = s.recv(200).decode("latin1"); s.close()
    return resp.split("\\r\\n")[0]
for t in {targets!r}:
    try:
        line = connect_via_proxy(t)
    except Exception as e:
        line = type(e).__name__
    (esc if " 200" in line else ok)(f"CONNECT {{t}} -> {{line}}")
s = socket.create_connection(("127.0.0.1", 3128), timeout=10)
s.sendall(b"GET http://example.com/ HTTP/1.1\\r\\nHost: example.com\\r\\n\\r\\n")
line = s.recv(200).decode("latin1").split("\\r\\n")[0]; s.close()
(esc if " 200" in line else ok)(f"GET http://example.com/ -> {{line}}")
'''


def probe_readonly(ctx):
    return r'''
for p in ("/opt/hyp/pwned", "/opt/hyp/bin/pwned", "/usr/local/bin/pwned", "/etc/pwned", "/pwned"):
    try:
        open(p, "w").write("x"); esc("wrote " + p)
    except Exception as e:
        ok(f"write {p}: {type(e).__name__}")
try:
    open("/work/ok.txt", "w").write("x"); os.remove("/work/ok.txt"); ok("/work is writable (expected)")
except Exception as e:
    esc(f"/work not writable: {e}")
'''


def probe_host(ctx):
    return r'''
for p in ("/var/run/docker.sock", "/run/docker.sock", "/run/containerd/containerd.sock"):
    if os.path.exists(p):
        try:
            s = socket.socket(socket.AF_UNIX); s.settimeout(2); s.connect(p); esc("connected " + p)
        except Exception as e:
            esc(f"{p} exists ({type(e).__name__})")
    else:
        ok(f"{p} absent")
pids = [d for d in os.listdir("/proc") if d.isdigit()]
cmds = []
for d in pids:
    try:
        cmds.append(open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode()[:80])
    except Exception:
        pass
hostish = [c for c in cmds if any(t in c for t in ("dockerd", "postgres", "systemd", "sshd", "tailscaled", "claude"))]
if len(pids) > 20 or hostish: esc(f"{len(pids)} pids visible, host-like: {hostish[:5]}")
else: ok(f"{len(pids)} pids visible, all container-local")
gpus = [d for d in os.listdir("/dev") if d.startswith("nvidia")]
(esc if gpus else ok)(f"gpu devices: {gpus}")
caps = [l for l in open("/proc/self/status") if l.startswith("CapEff")][0].split()[1]
(esc if int(caps, 16) else ok)(f"CapEff={caps}")
nnp = [l for l in open("/proc/self/status") if l.startswith("NoNewPrivs")][0].split()[1]
(ok if nnp == "1" else esc)(f"NoNewPrivs={nnp}")
ifaces = [l.split(":")[0].strip() for l in open("/proc/net/dev").readlines()[2:]]
(ok if ifaces == ["lo"] else esc)(f"interfaces: {ifaces}")
'''


def probe_transcripts(ctx):
    return f'''
import json
cfg = os.environ["CLAUDE_CONFIG_DIR"]
host_projects = set({ctx["host_projects"]!r})
mine = set(os.listdir(os.path.join(cfg, "projects")))
leak = mine & host_projects
(esc if leak else ok)(f"projects/: {{len(mine)}} entries, {{len(leak)}} shared with host")
h = os.path.join(cfg, "history.jsonl")
size = os.path.getsize(h) if os.path.exists(h) else 0
(esc if size else ok)(f"history.jsonl size {{size}}")
for sub in ("file-history", "sessions", "session-env", "shell-snapshots", "todos", "backups", "paste-cache", "jobs", "plugins", "skills"):
    p = os.path.join(cfg, sub)
    n = len(os.listdir(p)) if os.path.isdir(p) else 0
    (esc if n else ok)(f"{{sub}}/: {{n}} entries")
cj = json.load(open(os.path.join(cfg, ".claude.json")))
bad = [k for k in ("projects", "githubRepoPaths", "skillUsage", "pluginUsage") if k in cj]
(esc if bad else ok)(f".claude.json leaked keys: {{bad}}")
for f in walk(cfg):
    try:
        if {ctx["transcript_marker"]!r}.encode() in open(f, "rb").read():
            esc("host transcript marker found in " + f)
    except Exception:
        pass
ok("marker scan done")
'''


def probe_broker(ctx):
    return r'''
s = socket.socket(socket.AF_UNIX); s.settimeout(5)
try:
    s.connect("/run/hyp/broker.sock"); s.sendall(b"ping-scilearn\n")
    data = s.recv(100)
    if data.strip() == b"ping-scilearn": ok("broker socket reachable, echo matched")
    else: esc(f"broker echo mismatch: {data!r}")
except Exception as e:
    esc(f"broker socket NOT reachable: {type(e).__name__}")
'''


# ---------------------------------------------------------------- host-side

def test_proxy_policy() -> tuple[bool, list[str]]:
    """Allowlisted names that resolve to private space must be refused (stub resolver)."""
    out, good = [], True
    cases = {
        "127.0.0.1": False, "10.1.2.3": False, "172.17.0.1": False, "192.168.7.43": False,
        "100.85.63.24": False, "100.64.0.1": False, "169.254.169.254": False, "::1": False,
        "fd7a:115c:a1e0::1": False, "fe80::1": False, "::ffff:127.0.0.1": False, "0.0.0.0": False,
        "160.79.104.10": True,
    }
    for addr, expect in cases.items():
        res, reason, _ = proxy_mod.check_target("evil.anthropic.com", 443, lambda h, p, a=addr: [a])
        line = f"evil.anthropic.com -> {addr}: {'allow' if res else 'deny'} ({reason})"
        if res != expect:
            good = False
            line = "ESCAPED " + line
        out.append(line)
    res, reason, _ = proxy_mod.check_target("api.anthropic.com", 443, lambda h, p: ["160.79.104.10", "10.0.0.1"])
    out.append(f"mixed public+private -> {'allow' if res else 'deny'} ({reason})")
    good &= not res
    for host, port in (("example.com", 443), ("api.anthropic.com", 80), ("mcp-proxy.anthropic.com", 443),
                       ("anthropic.com.evil.net", 443), ("127.0.0.1", 443)):
        res, reason, _ = proxy_mod.check_target(host, port, lambda h, p: ["160.79.104.10"])
        out.append(f"{host}:{port} -> {'allow' if res else 'deny'} ({reason})")
        good &= not res
    return good, out


class EchoServer:
    def __init__(self, path: Path):
        self.path = path
        if path.exists() or path.is_socket():
            path.unlink()
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.bind(str(path))
        os.chmod(path, 0o600)
        self.sock.listen(8)
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def loop(self):
        while True:
            try:
                c, _ = self.sock.accept()
            except OSError:
                return
            try:
                c.settimeout(5)
                c.sendall(c.recv(100))
            except OSError:
                pass
            c.close()

    def close(self):
        self.sock.close()
        try:
            self.path.unlink()
        except OSError:
            pass


def main() -> int:
    SESSION.mkdir(parents=True, exist_ok=True)
    try:
        host_projects = sorted(os.listdir(ACCOUNT_DIR / "projects"))
    except OSError:
        host_projects = []
    # A marker planted into the host account dir's (masked) projects tree; must not be visible.
    marker = "SCILEARN_TRANSCRIPT_MARKER_51c2e0"
    mdir = ACCOUNT_DIR / "projects" / "-scilearn-escape-test-marker"
    ctx = {"host_ips": host_ips(), "ts_ips": tailscale_ips(), "pg_ips": pg_container_ips(),
           "host_projects": host_projects + ["-scilearn-escape-test-marker"], "transcript_marker": marker}
    print(f"context: host_ips={ctx['host_ips']} tailscale={ctx['ts_ips']} pg_container={ctx['pg_ips']}")

    config.PRIVATE.mkdir(parents=True, exist_ok=True)
    CANARY.write_bytes(b"PK\x03\x04" + CANARY_MARK.encode() + os.urandom(64))
    mdir.mkdir(exist_ok=True)
    (mdir / "marker.jsonl").write_text(json.dumps({"marker": marker}) + "\n")
    echo = EchoServer(SESSION / "broker.sock")

    tests = [
        ("read .env", probe_env), ("connect to postgres", probe_postgres),
        ("read label files", probe_labels), ("ssh to workers", probe_ssh),
        ("proxy refuses private/non-allowlisted", probe_proxy), ("read-only /opt/hyp and root", probe_readonly),
        ("docker.sock, host processes, caps", probe_host), ("other sessions' transcripts", probe_transcripts),
        ("broker socket reachable (positive control)", probe_broker),
    ]
    failures = 0
    try:
        good, lines = test_proxy_policy()
        print(f"{'PASS' if good else 'FAIL'}  proxy policy (host-side, stub resolver)")
        for ln in lines:
            print(f"      {ln}")
        failures += not good
        for name, fn in tests:
            good, lines = run_probe(fn.__name__, fn(ctx))
            print(f"{'PASS' if good else 'FAIL'}  {name}")
            for ln in lines:
                if ln != "PROBE_DONE":
                    print(f"      {ln}")
            failures += not good
    finally:
        echo.close()
        try:
            CANARY.unlink()
        except OSError:
            pass
        (mdir / "marker.jsonl").unlink(missing_ok=True)
        try:
            mdir.rmdir()
        except OSError:
            pass
        print(f"canary removed: {not CANARY.exists()}; marker removed: {not mdir.exists()}")
    print(f"\n{len(tests) + 1 - failures}/{len(tests) + 1} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
