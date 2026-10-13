#!/usr/bin/env python3
"""Launch a proposer session (or an arbitrary command) inside the scilearn sandbox.

Spec: paper/hypothesis_system/FRAMING.md 5.1 and 7.2; WAVE0_BRIEF items 3 and 10.

Session layout, /home/max/scilearn-sessions/<id>/:
  view/            -> /opt/hyp (read-only; filled by the view builder or --view-src)
  work/            -> /work (read-write, cwd)
  logs/            host only: proxy.jsonl, claude stream, stderr, launch records
  claude-projects/ -> <config>/projects (this session's transcripts)
  claude-state/    per-session stand-ins for every other entry of the account dir
  broker.sock      -> /run/hyp/broker.sock (owned by the harness broker, not us)
  proxy.sock       -> /run/hyp/proxy.sock (egress proxy started by this script)

The claude account dir is bind-mounted read-write so that .credentials.json stays
the one shared file (claude rewrites it by rename; a single-file bind would go stale).
Every other entry in it is masked by a per-session bind, so other sessions'
transcripts, history and settings are not visible.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from scilearn import config  # noqa: E402

IMAGE = config.SANDBOX_IMAGE
ACCOUNT_DIR = Path(os.environ.get("SCILEARN_CLAUDE_ACCOUNT", "/home/max/.claude-account-2"))
CLAUDE_LINK = Path("/home/max/.npm-global/bin/claude")
DEFAULT_MODEL = "claude-opus-5-5"

C_HOME = "/home/hyp"
C_CONFIG = f"{C_HOME}/.claude"
PROXY_PORT = 3128

# Names claude may read from the config dir that could steer a session. Masked even
# if absent on the host, so a proposer cannot plant them for other account sessions.
PREMASK_DIRS = ["agents", "commands", "output-styles", "hooks", "rules", "skills", "plugins",
                "projects", "todos", "statsig", "ide", "teams", "plans", "tasks"]
PREMASK_FILES = ["CLAUDE.md", "settings.json", "settings.local.json", "history.jsonl", ".claude.json"]
# Host-side placeholder content when a premasked file does not exist yet.
PLACEHOLDER = {"settings.local.json": "{}\n", "settings.json": "{}\n"}
SHARED = {".credentials.json"}

CLAUDE_JSON_KEEP = ["oauthAccount", "userID", "hasCompletedOnboarding", "lastOnboardingVersion",
                    "firstStartTime", "migrationVersion", "opusProMigrationComplete",
                    "sonnet1m45MigrationComplete", "installMethod", "hasSeenTasksHint",
                    "hasResetAutoModeOptInForDefaultOffer", "hasSeenAutoDefaultNotice"]
SESSION_SETTINGS = {
    "permissions": {"deny": ["WebFetch", "WebSearch"]},
    "skipDangerousModePermissionPrompt": True,
}


def session_paths(root: Path, sid: str) -> dict:
    d = root / sid
    return {
        "dir": d, "view": d / "view", "work": d / "work", "logs": d / "logs",
        "projects": d / "claude-projects", "state": d / "claude-state",
        "broker": d / "broker.sock", "proxy": d / "proxy.sock",
    }


def masked_entries() -> tuple[list[str], list[str]]:
    """(dirs, files) to mask inside the account dir: everything present plus PREMASK."""
    dirs, files = set(PREMASK_DIRS), set(PREMASK_FILES)
    for e in ACCOUNT_DIR.iterdir():
        if e.name in SHARED:
            continue
        (dirs if e.is_dir() and not e.is_symlink() else files).add(e.name)
    files -= dirs
    return sorted(dirs), sorted(files)


def seed_session(p: dict, view_src: str | None) -> None:
    p["dir"].mkdir(parents=True, exist_ok=True)
    os.chmod(p["dir"], 0o700)
    for k in ("view", "work", "logs", "projects", "state"):
        p[k].mkdir(exist_ok=True)
    if view_src:
        shutil.copytree(view_src, p["view"], dirs_exist_ok=True)
    dirs, files = masked_entries()
    for name in dirs:
        if name != "projects":
            (p["state"] / name).mkdir(exist_ok=True)
    for name in files:
        f = p["state"] / name
        if not f.exists():
            f.touch(mode=0o600)
            if name in PLACEHOLDER:
                f.write_text(PLACEHOLDER[name])
    cj = p["state"] / ".claude.json"
    if cj.stat().st_size == 0:
        try:
            src = json.loads((ACCOUNT_DIR / ".claude.json").read_text())
        except (OSError, ValueError):
            src = {}
        cj.write_text(json.dumps({k: src[k] for k in CLAUDE_JSON_KEEP if k in src}, indent=2))
    st = p["state"] / "settings.json"
    if st.read_text().strip() in ("", "{}"):
        st.write_text(json.dumps(SESSION_SETTINGS, indent=2))
    # Mount points that must exist on the host side so docker does not create root-owned ones.
    for name in dirs:
        t = ACCOUNT_DIR / name
        if not t.exists():
            t.mkdir()
    for name in files:
        t = ACCOUNT_DIR / name
        if not t.exists():
            t.touch(mode=0o600)
            if name in PLACEHOLDER:
                t.write_text(PLACEHOLDER[name])


def docker_cmd(p: dict, sid: str, inner: list[str], *, memory: str, cpus: str, pids: int,
               interactive: bool, broker: bool) -> list[str]:
    claude_bin = CLAUDE_LINK.resolve()
    name = f"scilearn-{sid}-{uuid.uuid4().hex[:6]}"
    proxy_url = f"http://127.0.0.1:{PROXY_PORT}"
    cmd = ["docker", "run", "--rm", "--name", name, "--label", f"scilearn.session={sid}",
           "--network", "none", "--user", "1000:1000", "--cap-drop", "ALL",
           "--security-opt", "no-new-privileges", "--read-only", "--init",
           "--pids-limit", str(pids), "--memory", memory, "--memory-swap", memory, "--cpus", cpus,
           "--hostname", "scilearn-sandbox",
           "--tmpfs", "/tmp:rw,nosuid,nodev,size=1g,mode=1777",
           "--tmpfs", f"{C_HOME}:rw,nosuid,nodev,size=1g,uid=1000,gid=1000,mode=0700",
           "-w", "/work"]
    if interactive:
        cmd.append("-i")

    def bind(src, dst, ro=False):
        cmd.extend(["--mount", f"type=bind,src={src},dst={dst}" + (",readonly" if ro else "")])

    bind(p["view"], "/opt/hyp", ro=True)
    bind(p["work"], "/work")
    if broker:
        bind(p["broker"], "/run/hyp/broker.sock")
    bind(p["proxy"], "/run/hyp/proxy.sock")
    bind(claude_bin, "/opt/claude/claude", ro=True)
    bind(ACCOUNT_DIR, C_CONFIG)
    dirs, files = masked_entries()
    for name in dirs:
        src = p["projects"] if name == "projects" else p["state"] / name
        bind(src, f"{C_CONFIG}/{name}")
    for name in files:
        bind(p["state"] / name, f"{C_CONFIG}/{name}")
    env = {
        "HOME": C_HOME, "CLAUDE_CONFIG_DIR": C_CONFIG,
        "PATH": "/opt/hyp/bin:/opt/claude:/usr/local/bin:/usr/bin:/bin",
        "HTTPS_PROXY": proxy_url, "HTTP_PROXY": proxy_url,
        "https_proxy": proxy_url, "http_proxy": proxy_url, "NO_PROXY": "", "no_proxy": "",
        "SCILEARN_PROXY_PORT": str(PROXY_PORT), "SCILEARN_SESSION": sid,
        "HYP_BROKER_SOCKET": "/run/hyp/broker.sock",
        "DISABLE_AUTOUPDATER": "1", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "DISABLE_ERROR_REPORTING": "1", "DISABLE_TELEMETRY": "1",
        "ENABLE_CLAUDEAI_MCP_SERVERS": "false",
    }
    for k, v in env.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.append(IMAGE)
    cmd.extend(inner)
    return cmd


def claude_inner(model: str, allow_web: bool, extra: list[str]) -> list[str]:
    inner = ["/opt/claude/claude", "-p", "--model", model, "--dangerously-skip-permissions",
             "--output-format", "stream-json", "--verbose", "--strict-mcp-config"]
    if not allow_web:
        inner += ["--disallowedTools", "WebFetch,WebSearch"]
    return inner + extra


def start_proxy(p: dict) -> subprocess.Popen:
    if p["proxy"].exists():
        p["proxy"].unlink()
    proc = subprocess.Popen(
        ["nice", "-n", "19", sys.executable, "-I", str(HERE / "proxy.py"),
         "--socket", str(p["proxy"]), "--log", str(p["logs"] / "proxy.jsonl")],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=open(p["logs"] / "proxy.stderr.log", "a"),
        start_new_session=True)
    for _ in range(100):
        if p["proxy"].exists():
            return proc
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    proc.terminate()
    raise SystemExit("proxy failed to start; see logs/proxy.stderr.log")


def audit_account_dir(before: set[str], p: dict) -> None:
    after = {e.name for e in ACCOUNT_DIR.iterdir()}
    new = sorted(after - before)
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "new_entries": new}
    with open(p["logs"] / "account_dir_audit.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    if new:
        print(f"warning: new entries in {ACCOUNT_DIR} during the run: {new}", file=sys.stderr)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_id")
    ap.add_argument("--prompt-file", help="prompt for claude -p (fed on stdin)")
    ap.add_argument("--cmd", help="run this bash command in the sandbox instead of claude")
    ap.add_argument("--view-src", help="copy this directory into <session>/view first")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--allow-web", action="store_true", help="do not disable WebFetch/WebSearch")
    ap.add_argument("--claude-arg", action="append", default=[], help="extra arg for claude (repeatable)")
    ap.add_argument("--sessions-root", default=str(config.SESSIONS))
    ap.add_argument("--memory", default="8g")
    ap.add_argument("--cpus", default="2")
    ap.add_argument("--pids", type=int, default=512)
    ap.add_argument("--no-broker-check", action="store_true", help="tests only: run without broker.sock")
    ap.add_argument("--dry-run", action="store_true", help="print the docker command and exit")
    a = ap.parse_args(argv)

    if not a.session_id.replace("-", "").replace("_", "").isalnum():
        ap.error("session_id must be alphanumeric with - or _")
    if bool(a.cmd) == bool(a.prompt_file):
        ap.error("give exactly one of --prompt-file or --cmd")
    p = session_paths(Path(a.sessions_root), a.session_id)
    broker = p["broker"].is_socket()
    if not broker and not a.no_broker_check and not a.dry_run:
        ap.error(f"broker socket missing: {p['broker']} (start the broker first)")

    inner = ["bash", "-c", a.cmd] if a.cmd else claude_inner(a.model, a.allow_web, a.claude_arg)
    cmd = docker_cmd(p, a.session_id, inner, memory=a.memory, cpus=a.cpus, pids=a.pids,
                     interactive=bool(a.prompt_file), broker=broker or a.dry_run)
    if a.dry_run:
        print(shlex.join(cmd))
        return 0

    seed_session(p, a.view_src)
    before = {e.name for e in ACCOUNT_DIR.iterdir()}
    proxy = start_proxy(p)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    with open(p["logs"] / "launches.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": stamp, "mode": "cmd" if a.cmd else "claude", "model": a.model,
                             "prompt_file": a.prompt_file, "docker": cmd}) + "\n")
    try:
        if a.cmd:
            rc = subprocess.run(cmd, stdin=subprocess.DEVNULL).returncode
        else:
            with open(a.prompt_file, "rb") as fin, \
                 open(p["logs"] / f"claude-{stamp}.stream.jsonl", "wb") as fout, \
                 open(p["logs"] / f"claude-{stamp}.stderr.log", "wb") as ferr:
                rc = subprocess.run(cmd, stdin=fin, stdout=fout, stderr=ferr).returncode
            print(f"exit {rc}; stream at {p['logs'] / f'claude-{stamp}.stream.jsonl'}")
    finally:
        proxy.terminate()
        try:
            proxy.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy.kill()
        audit_account_dir(before, p)
    return rc


if __name__ == "__main__":
    sys.exit(main())
