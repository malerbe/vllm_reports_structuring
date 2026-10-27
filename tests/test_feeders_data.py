from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from rsx.data import build_work_units, longest_first, select_units, shard_units, text_id
from rsx.feeders import (
    ReportFeeder,
    ReportUnit,
    available_feeders,
    make_feeder,
    parse_feeder_options,
)


def test_ct_rate_grouping(ct_rate_csv: Path):
    units = list(make_feeder("ct_rate", ct_rate_csv).iter_units())
    by_key = {u.key: u for u in units}
    assert by_key["valid_1_a"].members == ("valid_1_a_1.nii.gz", "valid_1_a_2.nii.gz")
    assert by_key["valid_2_a"].members == ("valid_2_a_1.nii.gz",)
    # different texts within one scan -> split
    assert {"valid_3_a#0", "valid_3_a#1"} <= set(by_key)
    text = by_key["valid_2_a"].text
    assert text.startswith("CLINICAL INFORMATION:\nCough.") and "FINDINGS:\nMild emphysema." in text
    # empty columns are skipped (no empty IMPRESSION heading)
    assert "IMPRESSION" not in by_key["valid_3_a#0"].text


def test_ct_rate_no_grouping(ct_rate_csv: Path):
    units = list(make_feeder("ct_rate", ct_rate_csv, {"group_reconstructions": False}).iter_units())
    assert len(units) == 6 and all(len(u.members) == 1 for u in units)


def test_ct_rate_missing_column(tmp_path: Path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"VolumeName": ["a_1.nii.gz"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing column"):
        list(make_feeder("ct_rate", p).iter_units())


def test_unknown_feeder_option(ct_rate_csv: Path):
    with pytest.raises(ValueError, match="unknown option"):
        make_feeder("ct_rate", ct_rate_csv, {"colums": []})


def test_generic_csv(tmp_path: Path):
    p = tmp_path / "r.csv"
    pd.DataFrame({"rid": ["1", "2"], "report": ["text a", ""], "extra": ["x", "y"]}).to_csv(
        p, index=False
    )
    feeder = make_feeder("csv", p, parse_feeder_options(["id_column=rid", "text_columns=[report]"]))
    units = list(feeder.iter_units())
    assert units == [ReportUnit(key="1", text="text a", members=("1",))]
    assert feeder.member_column == "rid"
    with pytest.raises(ValueError, match="needs"):
        list(make_feeder("csv", p).iter_units())


def test_registry_and_options():
    assert {"ct_rate", "csv"} <= set(available_feeders())
    assert issubclass(available_feeders()["ct_rate"], ReportFeeder)
    assert parse_feeder_options(["a=1", "b=[x, y]", "c=true", "d=text"]) == {
        "a": 1,
        "b": ["x", "y"],
        "c": True,
        "d": "text",
    }
    with pytest.raises(ValueError):
        parse_feeder_options(["novalue"])


def test_work_units_dedup(ct_rate_csv: Path):
    units = build_work_units(make_feeder("ct_rate", ct_rate_csv).iter_units())
    # valid_1_a and valid_4_b share the text -> one request, 3 members
    shared = [u for u in units if "valid_4_b" in u.keys]
    assert len(shared) == 1
    assert shared[0].keys == ("valid_1_a", "valid_4_b")
    assert set(shared[0].members) == {
        "valid_1_a_1.nii.gz",
        "valid_1_a_2.nii.gz",
        "valid_4_b_1.nii.gz",
    }
    assert shared[0].id == text_id(shared[0].text)
    assert len(units) == 4
    no_dedup = build_work_units(make_feeder("ct_rate", ct_rate_csv).iter_units(), text_dedup=False)
    assert len(no_dedup) == 5 and len({u.id for u in no_dedup}) == 5


def test_shard_and_select(ct_rate_csv: Path):
    units = build_work_units(make_feeder("ct_rate", ct_rate_csv).iter_units())
    shards = [shard_units(units, 3, i) for i in range(3)]
    assert sorted(u.id for s in shards for u in s) == sorted(u.id for u in units)
    assert select_units(units, sample=2, seed=1) == select_units(
        list(reversed(units)), sample=2, seed=1
    )
    assert len(select_units(units, limit=1)) == 1
    assert [u.keys for u in select_units(units, ids=["valid_2_a_1.nii.gz"])] == [("valid_2_a",)]
    lf = longest_first(units)
    assert len(lf[0].text) >= len(lf[-1].text)
    with pytest.raises(ValueError):
        shard_units(units, 2, 2)
