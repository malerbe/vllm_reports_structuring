"""Compare exported chest_ct_findings runs on CT-RATE: alert rates and label agreement.

  python scripts/evaluate_ctrate.py RUN [RUN ...] --labels ../valid_predicted_labels.csv
      [--mapping templates/chest_ct_findings/label_mapping.yaml] [--common] [--per-label]

For each run (a directory with exports/), prints:
  - the rate of each location alert column of the findings table (see the template postprocess)
  - agreement with the 18 CT-RATE labels per VolumeName: a label is predicted 1 when a finding of
    a mapped finding_type is present (variant "+uncertain": present or uncertain).
--common restricts every run to the VolumeNames extracted (status ok) by all runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
VARIANTS = {"present": {"present"}, "+uncertain": {"present", "uncertain"}}
ALERTS = ["lat_lobe_conflict", "site_lobe_conflict", "subset_of_invalid", "location_alert"]


def load(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    exports = run / "exports"
    reports = pd.read_parquet(exports / "reports.parquet")
    findings = pd.read_parquet(exports / "findings.parquet")
    return reports[reports["status"] == "ok"], findings


def alert_rates(findings: pd.DataFrame) -> dict[str, float]:
    items = findings.drop_duplicates(["unit_id", "item_idx"])
    out = {"findings": len(items)}
    for col in ALERTS:
        if col in items:
            out[col] = items[col].astype(bool).mean()
    if "site_unsupported" in items:
        out["site_unsupported"] = items["site_unsupported"].notna().mean()
    if "laterality_source" in items:
        lat = items["laterality_source"].dropna()
        for source in ("evidence", "sentence", "impression", "none"):
            out[f"laterality_from_{source}"] = (lat == source).mean() if len(lat) else float("nan")
    if "subset_of" in items:
        out["with_subset_of"] = items["subset_of"].notna().mean()
    return out


def predictions(
    findings: pd.DataFrame,
    members: list[str],
    mapping: dict[str, str],
    labels: list[str],
    statuses: set[str],
) -> pd.DataFrame:
    pred = pd.DataFrame(0, index=pd.Index(members, name="VolumeName"), columns=labels)
    hits = findings[findings["status"].isin(statuses)]
    for member, ftype in zip(hits["VolumeName"], hits["finding_type"], strict=True):
        label = mapping.get(ftype)
        if label in pred.columns and member in pred.index:
            pred.loc[member, label] = 1
    return pred


def agreement(pred: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    truth = truth.loc[pred.index, pred.columns]
    rows = {}
    for label in pred.columns:
        p, t = pred[label].astype(bool), truth[label].astype(bool)
        tp, fp, fn = int((p & t).sum()), int((p & ~t).sum()), int((~p & t).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
        rows[label] = {
            "pos": int(t.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    return pd.DataFrame(rows).T


def summary(table: pd.DataFrame, pred: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    tp, fp, fn = table["tp"].sum(), table["fp"].sum(), table["fn"].sum()
    exact = (pred.values == truth.loc[pred.index, pred.columns].values).all(axis=1).mean()
    return {
        "micro_f1": 2 * tp / (2 * tp + fp + fn),
        "macro_f1": table["f1"].mean(),
        "micro_precision": tp / (tp + fp),
        "micro_recall": tp / (tp + fn),
        "volumes_all_18_equal": exact,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument(
        "--mapping", type=Path, default=REPO / "templates/chest_ct_findings/label_mapping.yaml"
    )
    ap.add_argument("--common", action="store_true")
    ap.add_argument("--per-label", action="store_true")
    args = ap.parse_args()

    mapping = yaml.safe_load(args.mapping.read_text())
    truth = pd.read_csv(args.labels).set_index("VolumeName")
    labels = list(truth.columns)
    data = {f"{run.parent.name}/{run.name}": load(run) for run in args.runs}
    if args.common:
        common = set.intersection(*(set(r["VolumeName"]) for r, _ in data.values()))
        data = {
            k: (r[r["VolumeName"].isin(common)], f[f["VolumeName"].isin(common)])
            for k, (r, f) in data.items()
        }

    alerts, agree, per_label = {}, {}, {}
    for name, (reports, findings) in data.items():
        members = sorted(set(reports["VolumeName"]) & set(truth.index))
        alerts[name] = alert_rates(findings[findings["VolumeName"].isin(members)])
        for variant, statuses in VARIANTS.items():
            pred = predictions(findings, members, mapping, labels, statuses)
            table = agreement(pred, truth)
            agree[(name, variant)] = {"volumes": len(members), **summary(table, pred, truth)}
            if variant == "present":
                per_label[name] = table

    pd.set_option("display.width", 250)
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\n== Location alerts (share of findings)")
    print(pd.DataFrame(alerts).to_string())
    print("\n== Agreement with CT-RATE labels")
    print(pd.DataFrame(agree).to_string())
    if args.per_label:
        print("\n== F1 per label (present)")
        f1 = pd.DataFrame({n: t["f1"] for n, t in per_label.items()})
        f1.insert(0, "pos", next(iter(per_label.values()))["pos"])
        print(f1.to_string())
        for name, table in per_label.items():
            print(f"\n-- {name}")
            print(table.to_string())


if __name__ == "__main__":
    main()
