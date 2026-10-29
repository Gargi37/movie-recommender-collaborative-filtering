"""Measure real HTTP latency of the running Flask API (localhost, one process). Writes results/latency.json."""
import json, os, subprocess, sys, time, urllib.request
import numpy as np
PORT = 5077
proc = subprocess.Popen([sys.executable, '-W', 'ignore', 'api/app.py'], env={**os.environ, 'PORT': str(PORT)}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(60):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/health', timeout=1); break
        except Exception:
            time.sleep(0.5)
    users = json.load(urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/health'))['users']
    ids = sorted(int(u) for u in __import__('pandas').read_csv('data/processed/split/train_full.csv').user_id.unique())
    rng = np.random.default_rng(42); out = {}
    for label, qs in [('recommend k=10 (deployed)', 'k=10'), ('recommend k=10 diversify=safe', 'k=10&diversify=safe')]:
        t = []
        for u in rng.choice(ids, 300):
            s = time.perf_counter(); urllib.request.urlopen(f'http://127.0.0.1:{PORT}/api/recommend?user_id={u}&{qs}').read(); t.append((time.perf_counter() - s) * 1000)
        out[label] = dict(n=len(t), median_ms=float(np.median(t)), p95_ms=float(np.percentile(t, 95)), max_ms=float(np.max(t)))
    json.dump(out, open('results/latency.json', 'w'), indent=2); print(json.dumps(out, indent=2))
finally:
    proc.terminate()
