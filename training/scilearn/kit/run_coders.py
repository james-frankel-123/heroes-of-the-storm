"""Run three independent LLM codings of the blinded kit texts.

Each coder gets its own random ordering of hero codes (seeded, recorded), split into batches.
Every batch is a fresh `claude -p` call with no tools, no MCP servers, no session persistence,
run from an empty temporary directory, so no coder sees another coder's output or any repo file.
Responses are validated (all codes present, all 12 axes, integers in range); malformed
responses are retried with a fresh call.

Usage: nice -n 19 python3 -I run_coders.py [--coders 1,2,3] [--parallel 3]
Outputs: codings/prompts/*.txt, codings/coder<c>_batch<b>.json, codings/run_log.json
"""
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kitdefs import AXES, AXIS_RANGE  # noqa: E402

BLIND = os.path.join(HERE, "blind_text")
OUT = os.path.join(HERE, "codings")
MODEL = "claude-opus-5-5"
CONFIG_DIR = os.path.expanduser("~/.claude-account-2")
CODER_SEEDS = {1: 5701, 2: 5702, 3: 5703}
BATCH = 13
MAX_TRIES = 4

INSTRUCTIONS = """You are coding hero kits for a research table. Below is a rubric, then {n} hero texts.

The texts are anonymized on purpose: names of the hero, its abilities, talents and summons were replaced by neutral tokens. Do not try to identify any hero, and do not use anything you know about specific heroes. Code each text only from what it says, under the rubric. Do not judge strength, balance, win rate, tier or meta.

Code every hero code listed: {codes}.
Return strict JSON only (one object keyed by hero code), no prose and no code fences.

===== RUBRIC =====
{rubric}

===== HERO TEXTS =====
{texts}
"""


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def validate(obj, codes):
    if not isinstance(obj, dict):
        return "not an object"
    missing = [c for c in codes if c not in obj]
    if missing:
        return f"missing codes {missing}"
    for c in codes:
        v = obj[c]
        for a in AXES:
            x = v.get(a)
            lo, hi = AXIS_RANGE[a]
            if not isinstance(x, int) or isinstance(x, bool) or not lo <= x <= hi:
                return f"{c}.{a}={x!r} invalid"
    return None


def parse(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    a, b = t.find("{"), t.rfind("}")
    return json.loads(t[a:b + 1])


def call(prompt):
    with tempfile.TemporaryDirectory(prefix="kitcoder_") as cwd:
        env = dict(os.environ, CLAUDE_CONFIG_DIR=CONFIG_DIR)
        cmd = ["nice", "-n", "19", "claude", "-p", "--model", MODEL, "--tools", "", "--strict-mcp-config",
               "--no-session-persistence", "--output-format", "json"]
        r = subprocess.run(cmd, input=prompt, capture_output=True, text=True, cwd=cwd, env=env, timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"claude exit {r.returncode}: {r.stderr[-500:]}")
    return json.loads(r.stdout)


def run_batch(coder, b, codes, prompt, prompt_path):
    out_path = os.path.join(OUT, f"coder{coder}_batch{b}.json")
    if os.path.exists(out_path):
        return out_path, "cached"
    attempts = []
    for k in range(1, MAX_TRIES + 1):
        started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            meta = call(prompt)
            obj = parse(meta.get("result", ""))
            err = validate(obj, codes)
        except Exception as e:  # malformed output or CLI failure: retry fresh
            meta, obj, err = {}, None, f"{type(e).__name__}: {e}"
        attempts.append({"attempt": k, "started": started, "error": err,
                         "models": sorted((meta.get("modelUsage") or {}).keys())})
        if err is None:
            rec = {"coder": coder, "batch": b, "codes": codes, "prompt_file": os.path.relpath(prompt_path, HERE),
                   "prompt_sha256": sha(prompt), "model_requested": MODEL,
                   "models_reported": sorted((meta.get("modelUsage") or {}).keys()),
                   "session_id": meta.get("session_id"), "started": started,
                   "finished": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "attempts": attempts, "raw_result": meta.get("result"), "codes_out": obj}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=1, ensure_ascii=False)
            return out_path, f"ok after {k}"
        time.sleep(5)
    raise RuntimeError(f"coder{coder} batch{b} failed: {attempts}")


def main():
    coders = [1, 2, 3]
    par = 3
    for i, a in enumerate(sys.argv):
        if a == "--coders":
            coders = [int(x) for x in sys.argv[i + 1].split(",")]
        if a == "--parallel":
            par = int(sys.argv[i + 1])
    os.makedirs(os.path.join(OUT, "prompts"), exist_ok=True)
    rubric = open(os.path.join(HERE, "RUBRIC.md"), encoding="utf-8").read()
    texts = {f[:-4]: open(os.path.join(BLIND, f), encoding="utf-8").read().strip()
             for f in sorted(os.listdir(BLIND)) if f.endswith(".txt")}
    jobs = []
    for c in coders:
        order = sorted(texts)
        random.Random(CODER_SEEDS[c]).shuffle(order)
        nb = -(-len(order) // BATCH)
        for b in range(nb):
            codes = order[b * BATCH:(b + 1) * BATCH]
            body = "\n\n".join(f"----- {code} -----\n{texts[code]}" for code in codes)
            prompt = INSTRUCTIONS.format(n=len(codes), codes=", ".join(codes), rubric=rubric, texts=body)
            p = os.path.join(OUT, "prompts", f"coder{c}_batch{b}.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(prompt)
            jobs.append((c, b, codes, prompt, p))
    log = {"model": MODEL, "coder_seeds": CODER_SEEDS, "batch_size": BATCH,
           "template_sha256": sha(INSTRUCTIONS), "rubric_sha256": sha(rubric),
           "cli": "CLAUDE_CONFIG_DIR=~/.claude-account-2 claude -p --model claude-opus-5-5 --tools '' "
                  "--strict-mcp-config --no-session-persistence --output-format json (empty temp cwd)",
           "started": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "results": []}
    with cf.ThreadPoolExecutor(par) as ex:
        futs = {ex.submit(run_batch, *j): j[:2] for j in jobs}
        for fu in cf.as_completed(futs):
            c, b = futs[fu]
            path, status = fu.result()
            print(f"coder{c} batch{b}: {status}", flush=True)
            log["results"].append({"coder": c, "batch": b, "status": status})
    log["finished"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(os.path.join(OUT, "run_log.json"), "w") as f:
        json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
