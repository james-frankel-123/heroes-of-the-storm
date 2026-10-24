"""Manifest fail-closed checks (C7): missing or empty manifest, zero matching
entries, missing or changed files, incomplete inventories, code bundle pins."""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(HERE, "p3_manifest.py")


def run(*args, root):
    return subprocess.run([sys.executable, TOOL, *args, "--root", root], capture_output=True, text=True).returncode


def test_manifest():
    with tempfile.TemporaryDirectory() as td:
        os.makedirs(os.path.join(td, "personalization", "remote"))
        for f, txt in (("personalization/a.py", "x = 1\n"), ("personalization/remote/b.sh", "echo\n"),
                       ("shared.py", "y = 2\n"), ("data.bin", "abc")):
            open(os.path.join(td, f), "w").write(txt)
        man = os.path.join(td, "m.json")
        assert run("verify", man, root=td) == 1                                   # missing manifest
        json.dump({"schema": 1, "files": []}, open(man, "w"))
        assert run("verify", man, root=td) == 1                                   # empty manifest
        cwd = os.getcwd()
        os.chdir(td)
        try:
            assert run("add", man, "data.bin", "--role", "upstream", "--stage", "window-cache", "--host", "h1",
                       root=td) == 0
            open("empty.bin", "w").close()
            assert run("add", man, "empty.bin", "--role", "upstream", "--stage", "x", root=td) == 1
            assert run("add", man, "nope.bin", "--role", "upstream", "--stage", "x", root=td) == 1
        finally:
            os.chdir(cwd)
        e = json.load(open(man))["files"][0]
        assert e["git"] and (e["git"].startswith("src:") or len(e["git"]) >= 7)
        assert run("verify", man, root=td) == 0
        assert run("verify", man, "--role", "tournament", root=td) == 1          # no matching entries
        assert run("verify", man, "--role", "upstream", "--complete", root=td) == 0
        assert run("verify", man, "--role", "personalization,upstream", "--complete", root=td) == 1
        assert subprocess.run([sys.executable, TOOL, "code", man, "--pass", "p1", "--host", "h1", "--root", td],
                              capture_output=True).returncode == 0
        assert run("verify", man, "--host", "h1", root=td) == 0
        open(os.path.join(td, "personalization/a.py"), "a").write("z = 3\n")
        assert run("verify", man, "--host", "h1", root=td) == 1                  # code changed
        assert run("verify", man, "--host", "h2", root=td) == 0                  # other host: skipped
        open(os.path.join(td, "data.bin"), "a").write("d")
        assert run("verify", man, root=td) == 1                                   # data changed


if __name__ == "__main__":
    test_manifest()
    print("manifest tests pass")
