"""Optional real-data check (skipped by default; run with `pytest --run-network`)."""

import pytest

from clinical_tabred.download import download_component


@pytest.mark.network
def test_download_one_real_file(real_config, tmp_path):
    cycle = real_config["cycles"][0]
    row = download_component(cycle, "DIQ", real_config["download"], tmp_path, {})
    assert row["validation"] == "ok"
    assert row["bytes"] > 100_000
