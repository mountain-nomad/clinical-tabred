"""Build the standalone, submitted Jupyter notebook from this package's source code.

The notebook must run on its own (no repository, no package, no config file), so
the real source of every function is copied into code cells with `inspect`.
Markdown text comes from `notebooks/templates/*.md`. A template may contain:

    <!-- source: prepare.read_xpt prepare.merge_cycle -->   -> code cell with that source
    ```python ... ```                                       -> code cell (short calls)
    <!-- setup --> / <!-- imports --> / <!-- config -->     -> generated cells

Run `python -m clinical_tabred.run --stage notebook` after any code change so the
notebook and the repository always contain the same logic.
"""

import ast
import importlib
import inspect
import pprint
import re
from pathlib import Path

import nbformat

MODULES = ["download", "prepare", "splits", "models", "evaluation", "experiment", "benchmark",
           "drift", "plotting", "summaries"]
# Packages the notebook needs: import name -> pip name.
PACKAGES = {"pandas": "pandas", "numpy": "numpy", "sklearn": "scikit-learn",
            "xgboost": "xgboost", "lightgbm": "lightgbm", "scipy": "scipy",
            "matplotlib": "matplotlib", "requests": "requests"}
TOKEN = re.compile(r"<!--\s*(source|setup|imports|config)\s*:?\s*(.*?)-->|```python\n(.*?)```",
                   re.DOTALL)


def assigned_names(node: ast.Assign) -> list[str]:
    """Names on the left of `a = ...` or `a, b = ...`."""
    names = []
    for target in node.targets:
        elements = target.elts if isinstance(target, ast.Tuple) else [target]
        names += [element.id for element in elements if isinstance(element, ast.Name)]
    return names


def object_source(module_name: str, name: str) -> str:
    """Source code of a function or a module-level constant of clinical_tabred.<module>."""
    module = importlib.import_module(f"clinical_tabred.{module_name}")
    obj = getattr(module, name)
    if inspect.isfunction(obj):
        return inspect.getsource(obj).rstrip()
    text = inspect.getsource(module)
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and name in assigned_names(node):
            return ast.get_source_segment(text, node)
    raise ValueError(f"{name} not found in clinical_tabred.{module_name}")


def imports_cell() -> str:
    """All imports used by the copied modules, merged per module (no package imports)."""
    plain, from_imports = set(), {}
    for module_name in MODULES:
        text = inspect.getsource(importlib.import_module(f"clinical_tabred.{module_name}"))
        for node in ast.parse(text).body:
            if isinstance(node, ast.Import):
                plain.add(ast.get_source_segment(text, node))
            elif isinstance(node, ast.ImportFrom) and node.module != "clinical_tabred" \
                    and not node.module.startswith("clinical_tabred."):
                from_imports.setdefault(node.module, set()).update(a.name for a in node.names)
    lines = sorted(plain) + [f"from {module} import {', '.join(sorted(names))}"
                             for module, names in sorted(from_imports.items())]
    return ("%matplotlib inline\n# Imports used by the functions below (collected from the "
            "project modules).\n" + "\n".join(lines) +
            "\nfrom IPython.display import Markdown, display\n\n"
            'logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")\n'
            'logger = logging.getLogger("clinical_tabred")\n'
            'pd.set_option("display.max_columns", 40)\npd.set_option("display.width", 160)')


def setup_cell(versions: dict[str, str]) -> str:
    """Cell that installs only missing packages (pinned to the tested versions)."""
    required = {module: f"{pip}=={versions[pip]}" for module, pip in PACKAGES.items()}
    return f'''# Check the required packages and install ONLY the missing ones (tested versions).
import importlib.util
import sys

REQUIRED = {pprint.pformat(required, sort_dicts=False)}
missing = [req for module, req in REQUIRED.items() if importlib.util.find_spec(module) is None]
if missing:
    to_install = " ".join(missing)
    print("Installing missing packages:", to_install)
    %pip install -q {{to_install}}
else:
    print("All required packages are already installed.")

from importlib.metadata import version
print("Python", sys.version.split()[0])
for module, requirement in REQUIRED.items():
    package = requirement.split("==")[0]
    print(f"{{package:14s}} installed {{version(package):10s}} (tested with {{requirement.split('==')[1]}})")
# The notebook is self-contained: the project package is NOT needed (expected: False).
print("Project package importable:", importlib.util.find_spec("clinical_tabred") is not None)'''


def config_cell(config: dict) -> str:
    """The CONFIG dictionary (from configs/experiment.yaml) with notebook-relative paths."""
    config = {**config, "paths": {"raw_dir": "nhanes_data",
                                  "processed_dir": "notebook_outputs/processed",
                                  "results_dir": "notebook_outputs",
                                  "figures_dir": "notebook_outputs/figures"}}
    return ("# All settings in one place (generated from the project configuration file).\n"
            '# RUN_MODE = "quick" runs a small smoke test; "full" reproduces the reported results.\n'
            'RUN_MODE = "full"\n\n'
            f"CONFIG = {pprint.pformat(config, sort_dicts=False, width=95)}\n\n"
            f"{object_source('benchmark', 'quick_config')}\n\n\n"
            'if RUN_MODE == "quick":\n    CONFIG = quick_config(CONFIG)\n\n'
            'RAW_DIR = Path(CONFIG["paths"]["raw_dir"])\n'
            'PROCESSED_DIR = Path(CONFIG["paths"]["processed_dir"])\n'
            'RESULTS_DIR = Path(CONFIG["paths"]["results_dir"])\n'
            'FIGURES_DIR = Path(CONFIG["paths"]["figures_dir"])\n'
            "for folder in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR]:\n"
            "    folder.mkdir(parents=True, exist_ok=True)\n"
            'print("Run mode:", RUN_MODE, "| seed:", CONFIG["seed"])')


def template_cells(text: str, config: dict, versions: dict) -> list:
    """Turn one template into a list of notebook cells (markdown and code)."""
    cells, position = [], 0
    for match in TOKEN.finditer(text):
        markdown = text[position:match.start()].strip()
        if markdown:
            cells.append(nbformat.v4.new_markdown_cell(markdown))
        kind, argument, code = match.groups()
        if code is not None:
            cells.append(nbformat.v4.new_code_cell(code.strip()))
        elif kind == "source":
            names = [item.split(".") for item in argument.split()]
            body = "\n\n\n".join(object_source(module, name) for module, name in names)
            cells.append(nbformat.v4.new_code_cell(body))
        else:
            generated = {"setup": lambda: setup_cell(versions), "imports": imports_cell,
                         "config": lambda: config_cell(config)}[kind]()
            cells.append(nbformat.v4.new_code_cell(generated))
        position = match.end()
    if text[position:].strip():
        cells.append(nbformat.v4.new_markdown_cell(text[position:].strip()))
    return cells


def build_notebook(config: dict, template_dir: Path, output_path: Path,
                   versions: dict[str, str]) -> Path:
    """Assemble all templates (in file-name order) into one notebook file."""
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3",
                                       "language": "python"}
    for template in sorted(Path(template_dir).glob("*.md")):
        notebook.cells.extend(template_cells(template.read_text(), config, versions))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, output_path)
    return Path(output_path)
