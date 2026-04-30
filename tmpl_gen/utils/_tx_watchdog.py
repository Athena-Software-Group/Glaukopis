#!/usr/bin/env python
"""Poll Neo4j every poll_secs; terminate any non-system tx exceeding max_secs."""
from neo4j import GraphDatabase
import json, time, re, sys

cfg = json.load(open("tmpl_gen/data_generation/neo4j-local-config.json"))
drv = GraphDatabase.driver(cfg["uri"], auth=tuple(cfg["auth"]))

MAX_SECS  = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
POLL_SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
RUN_SECS  = float(sys.argv[3]) if len(sys.argv) > 3 else 7200.0

def parse_iso_duration(s: str) -> float:
    # Neo4j returns durations like "PT16M38.469S"
    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:([\d.]+)S)?$", s)
    if not m:
        return 0.0
    h, mi, sec = m.groups()
    return (float(h or 0) * 3600 + float(mi or 0) * 60 + float(sec or 0))

t_start = time.time()
print(f"watchdog: max_secs={MAX_SECS} poll_secs={POLL_SECS} run_secs={RUN_SECS}")
killed = 0
while time.time() - t_start < RUN_SECS:
    try:
        with drv.session(database=cfg["db_name"]) as s:
            txs = list(s.run("SHOW TRANSACTIONS").data())
            for t in txs:
                cq = str(t.get('currentQuery', ''))
                if 'SHOW TRANSACTIONS' in cq:
                    continue
                elapsed = parse_iso_duration(str(t.get('elapsedTime', 'PT0S')))
                if elapsed > MAX_SECS:
                    tid = t.get('transactionId')
                    print(f"  [{time.strftime('%H:%M:%S')}] killing {tid} elapsed={elapsed:.1f}s q={cq[:100]}")
                    try:
                        s.run(f"TERMINATE TRANSACTION '{tid}'")
                        killed += 1
                    except Exception as e:
                        print(f"    failed to kill: {e}")
    except Exception as e:
        print(f"  watchdog error: {type(e).__name__}: {e}")
    time.sleep(POLL_SECS)

print(f"watchdog exiting after {time.time()-t_start:.0f}s; killed={killed}")
drv.close()
