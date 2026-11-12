"""Checks of the generated presentation pack (run after --stage presentation)."""

import json
import math
import re

import pytest
from conftest import REPO

from clinical_tabred.summaries import lookup

PACK = REPO / "for_presentation"


@pytest.fixture(scope="module")
def pack():
    if not (PACK / "slides.md").exists():
        pytest.skip("presentation pack not built yet (run --stage presentation)")
    return PACK


def test_key_numbers_match_result_files(pack):
    entries = json.loads((pack / "key_numbers.json").read_text())
    assert entries
    for entry in entries:
        source = entry["source"].removeprefix("results/")
        expected = lookup(REPO / "results", source, entry["column"], entry.get("filter"),
                          entry.get("agg", "single"))
        if isinstance(expected, str):
            assert entry["value"] == expected, entry["id"]
        else:
            assert math.isclose(entry["value"], expected, rel_tol=1e-9), entry["id"]


def test_every_referenced_file_exists(pack):
    text = (pack / "slides.md").read_text()
    references = []
    for line in text.splitlines():
        if line.startswith("Visuals:"):
            references += [p.strip() for p in line.removeprefix("Visuals:").split("|")]
    references += [f"reference/sample_slides/{m}" for m in re.findall(r"sample-\d\d\.png", text)]
    missing = [r for r in references if r and r != "none" and not (pack / r).exists()]
    assert not missing, missing
    assert text.count("## Slide ") == 28


def test_no_sample_deck_leftovers(pack):
    for path in pack.rglob("*"):
        if path.suffix in {".md", ".csv", ".json"} and "reference" not in path.parts:
            text = path.read_text().lower()
            for word in ["spam", "enron", "illustrative"]:
                assert word not in text, (path, word)
            assert not re.search(r"\bp00\b", text), path
