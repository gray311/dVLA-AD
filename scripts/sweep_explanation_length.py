"""With steps=16/rep=4 fixed (the quality config from the steps/rep sweep),
vary the explanation slot LENGTH to see if a shorter slot cuts the tail-bleed
(field names / trajectory numbers leaking into the prose) without losing the
coherent 2-3 sentence content.

Length is set via env DVLA_N_EXPLANATION_TOKENS, read by template_v3 at import.
The engine is reloaded per length because the template scaffold changes.
"""
import json
import os
import subprocess
import sys

ROOT = "/weka/home/ext-yingzima/dVLA-AD-ad4fcc21"
LENGTHS = [100, 72, 56]


def main():
    out = {}
    for n in LENGTHS:
        env = dict(os.environ, DVLA_N_EXPLANATION_TOKENS=str(n))
        # Run the per-length worker in a fresh process so the module-level
        # constant + engine are rebuilt cleanly.
        res = subprocess.run(
            [sys.executable, f"{ROOT}/scripts/_explen_worker.py"],
            env=env, capture_output=True, text=True,
        )
        print(res.stdout[-4000:])
        if res.returncode != 0:
            print("STDERR:", res.stderr[-2000:])


if __name__ == "__main__":
    main()
