"""The submitted notebook must match the repository code and contain the required sections."""
import json

import nbformat
from build_notebook import OUT, build
from conftest import REPO

SECTIONS = ["## 1. Project introduction", "## 2. Dataset description", "## 3. Data loading",
            "## 4. Data preprocessing", "## 5. Exploratory data analysis", "## 6. Train/test splitting",
            "## 7. Cross-validation", "## 8. Model training", "## 9. Model evaluation",
            "## 10. Temporal drift analysis", "## 11. Results and comparison", "## 12. Limitations and conclusion"]


def test_notebook_is_up_to_date_with_the_code():
    saved = nbformat.read(OUT, 4)
    fresh = build()
    assert [c.source for c in saved.cells] == [c.source for c in fresh.cells], \
        "notebook is stale: run python scripts/build_notebook.py and execute it"


def test_notebook_has_all_sections_and_ran_without_errors():
    nb = nbformat.read(OUT, 4)
    text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    for section in SECTIONS:
        assert section in text
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert all(c.execution_count for c in code), "notebook is not fully executed"
    assert not [o for c in code for o in c.outputs if o.output_type == "error"]
    assert OUT.name == json.loads((REPO / "configs" / "experiment.yaml").read_text().split("notebook_name: ")[1]
                                  .split("\n")[0])
