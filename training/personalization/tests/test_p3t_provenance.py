"""P3 tournament: per-pass vintage table and draft provenance (p3t_systems).
  V1 P3T_VINTAGES points the vintage table at a per-pass copy made from the committed template;
     writes go to the copy, never the template
  V2 provenance() records host, kernel and code hashes and the shard, as JSON
Run: python3 personalization/tests/test_p3t_provenance.py"""
import json
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="p3t_prov_")
os.environ["P3_CACHE"] = os.path.join(TMP, "cache")
os.environ["P3_RESULTS"] = os.path.join(TMP, "results")
os.environ["P3T_VINTAGES"] = os.path.join(TMP, "results", "pass", "p3t_vintages.json")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import p3t_systems as SY


def main():
    assert SY.VFILE == os.environ["P3T_VINTAGES"] and os.path.exists(SY.VFILE)
    tmpl = open(SY.VTEMPLATE).read()
    assert open(SY.VFILE).read() == tmpl
    vs = json.load(open(SY.VFILE))
    k = next(iter(vs))
    vs[k]["combiner_note"] = "test write"
    json.dump(vs, open(SY.VFILE, "w"))
    assert open(SY.VTEMPLATE).read() == tmpl
    print("V1 ok")
    p = json.loads(SY.provenance("1/2"))
    assert p["shard"] == "1/2" and p["host"] and len(p["code_sha256"]) == 16 and p["vintages_file"] == SY.VFILE
    print("V2 ok", {k_: v_ for k_, v_ in p.items() if not k_.startswith("kernel:")})
    print("ALL OK")


if __name__ == "__main__":
    main()
