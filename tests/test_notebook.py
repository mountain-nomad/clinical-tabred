"""Static checks that the submitted notebook is standalone (no package, no repo paths)."""

import json
import re

import pytest
from conftest import REPO

NOTEBOOK = REPO / "notebooks" / "CSCI447P034.ipynb"
REPO_PATH = re.compile(r"""["'](\.\./)?(configs|data/processed|data/raw|results|figures|src)(/|["'])""")


@pytest.fixture(scope="module")
def code_cells() -> list[str]:
    if not NOTEBOOK.exists():
        pytest.skip("notebook not built yet (run --stage notebook)")
    notebook = json.loads(NOTEBOOK.read_text())
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_notebook_does_not_import_project_package(code_cells):
    for source in code_cells:
        assert "import clinical_tabred" not in source
        assert "from clinical_tabred" not in source


def test_notebook_does_not_read_repository_paths(code_cells):
    for source in code_cells:
        assert not REPO_PATH.search(source), REPO_PATH.search(source).group(0)
        assert "experiment.yaml" not in source


def test_notebook_defines_config_and_downloads_data(code_cells):
    joined = "\n".join(code_cells)
    assert "CONFIG = {" in joined
    assert "download_all(CONFIG, RAW_DIR)" in joined
    assert "'raw_dir': 'nhanes_data'" in joined
