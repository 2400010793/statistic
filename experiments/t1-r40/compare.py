"""Print the official validation history and paired E9.25 changes from saved records."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
with (ROOT / "p2_scores.csv").open() as stream:
    scores = list(csv.DictReader(stream))
print("P2 official validation (higher is better)")
for row in sorted(scores, key=lambda r: float(r["headline"]), reverse=True):
    print(f"{float(row['headline']):7.4f}  {row['model']}")

paired = json.loads((ROOT / "e925_paired_metrics.json").read_text())
for cohort, records in paired["cohorts"].items():
    print(f"\nE9.25 {cohort}: equal-size R40 -> carry, six paired batches")
    for metric, record in records["summary"].items():
        print(f"{metric:14} {record['matched_r40']:.6f} -> {record['carry']:.6f}"
              f"  improved {record['better_batches']}/6")
