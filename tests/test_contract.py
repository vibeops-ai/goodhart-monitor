"""The contract must fail loudly. A silent coercion produces an authoritative
record that is wrong, which is the exact failure this company sells against."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from goodhart_monitor.contract import ContractError, validate


def base(**over):
    df = pd.DataFrame({
        "entity_id": ["a", "a", "b", "b"],
        "t": [1, 2, 1, 2],
        "score": [0.1, 0.9, 0.2, 0.3],
        "label": [0, 1, 0, 0],
    })
    for k, v in over.items():
        df[k] = v
    return df


def test_accepts_a_well_formed_stream():
    s = validate(base())
    assert s.n_rows == 4 and s.n_entities == 2
    assert s.has_time and s.has_onset


def test_infers_onset_from_time_varying_label():
    s = validate(base())
    assert s.df.loc[s.df.entity_id == "a", "onset_t"].iloc[0] == 2
    assert pd.isna(s.df.loc[s.df.entity_id == "b", "onset_t"].iloc[0])


def test_does_not_invent_onset_for_entity_level_labels():
    """A label repeated on every row has no onset; inferring one would fabricate
    a lead time out of nothing."""
    df = base()
    df["label"] = [1, 1, 0, 0]          # constant within each entity
    s = validate(df)
    assert not s.has_onset


@pytest.mark.parametrize("mutate,fragment", [
    (lambda d: d.drop(columns=["score"]), "missing required column"),
    (lambda d: d.assign(score=[0.1, None, 0.2, 0.3]), "missing value"),
    (lambda d: d.assign(score=[0.1, np.inf, 0.2, 0.3]), "inf"),
    (lambda d: d.assign(label=[0, 2, 0, 0]), "must be 0/1"),
    (lambda d: d.assign(label=[0, None, 0, 0]), "unadjudicated"),
    (lambda d: d.assign(label=[0, 0, 0, 0]), "single class"),
    (lambda d: d.assign(score=["a", "b", "c", "d"]), "must be numeric"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
    (lambda d: d.iloc[0:0], "empty"),
])
def test_rejects_and_explains(mutate, fragment):
    with pytest.raises(ContractError) as e:
        validate(mutate(base()))
    assert fragment in str(e.value).lower()


def test_flat_stream_without_time_is_valid(flat_stream):
    assert not flat_stream.has_time
    assert not flat_stream.has_onset


def test_flat_stream_rejects_duplicate_entities():
    df = pd.DataFrame({"entity_id": ["a", "a"], "score": [0.1, 0.2], "label": [0, 1]})
    with pytest.raises(ContractError) as e:
        validate(df)
    assert "duplicate entity_id" in str(e.value)


def test_rows_are_sorted_deterministically():
    df = base().iloc[::-1].reset_index(drop=True)
    s = validate(df)
    assert list(s.df["t"]) == [1, 2, 1, 2]
    assert list(s.df["entity_id"]) == ["a", "a", "b", "b"]


def test_subgroup_candidates_exclude_reserved():
    s = validate(base(Age=[50, 50, 60, 60]))
    assert "Age" in s.subgroup_candidates()
    assert not {"score", "label", "t", "entity_id"} & set(s.subgroup_candidates())


def test_content_hash_survives_a_writer_change():
    """A parquet writer upgrade changes the file bytes and must not read as a
    finding. Hashing the file made a library version a difference in the
    record."""
    import hashlib, tempfile, pathlib
    from goodhart_monitor.contract import content_sha256, load
    df = base()
    files, contents = set(), set()
    with tempfile.TemporaryDirectory() as d:
        for name, kw in [("a", {"compression": "snappy"}),
                         ("b", {"compression": "gzip"}),
                         ("c", {"compression": None}),
                         ("d", {"compression": "snappy", "row_group_size": 2})]:
            p = pathlib.Path(d) / f"{name}.parquet"
            df.to_parquet(p, index=False, **kw)
            files.add(hashlib.sha256(p.read_bytes()).hexdigest())
            contents.add(content_sha256(load(p)))
    assert len(files) > 1, "expected the writer settings to change the bytes"
    assert len(contents) == 1, "content hash must not move with the container"


def test_content_hash_ignores_row_order():
    from goodhart_monitor.contract import content_sha256
    df = base()
    assert content_sha256(validate(df)) == content_sha256(validate(df.iloc[::-1]))


def test_content_hash_changes_when_a_value_changes():
    from goodhart_monitor.contract import content_sha256
    df = base()
    before = content_sha256(validate(df))
    df.loc[0, "score"] = 0.99
    assert content_sha256(validate(df)) != before
