"""Text files of the presentation pack: algorithm, references, style guide, README, placeholders,
and rendering of the sample deck to PNG."""

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

from pptx import Presentation

ALGORITHM_LINES = [
    "Input: train, validation and test splits; grid G_m for each model m; seed s",
    "Output: tuned models, test metrics with 95% CIs, ablation table",
    "for each model m in {LR, RF, XGBoost, LightGBM, MLP} do",
    "    for each configuration θ in G_m do",
    "        fit ColumnTransformer on the training split only",
    "        if m = MLP then",
    "            for epoch e = 1 … 200: partial_fit; record train/val loss and accuracy",
    "            keep weights of the epoch with lowest validation loss; stop after 10 epochs without improvement",
    "        else fit m_θ on the training split",
    "        score validation PR-AUC (ROC-AUC breaks ties)",
    "    θ* ← best configuration of m",
    "    τ* ← threshold that maximises F1 on the validation split",
    "    evaluate m_θ* on the test split at τ = 0.5 and τ = τ*",
    "    95% bootstrap CIs of ROC-AUC and PR-AUC (1000 resamples)",
    "select the best model on validation; paired bootstrap against the others",
    "ablation: retrain the best model with one change at a time (same split and seed)",
]
ALGORITHM_GROUPS = [
    ("Lines 1–2", "Inputs and outputs: three time-ordered splits, small grids and a fixed seed."),
    ("Lines 3–5", ("Loop over models and grid configurations; preprocessing is fitted on training "
                  "data only to avoid leakage.")),
    ("Lines 6–9", ("The MLP is trained one epoch at a time with early stopping; other models are "
                  "fitted in one call.")),
    ("Lines 10–12", "Model and threshold selection use validation data only, never the test set."),
    ("Lines 13–16", "Single test evaluation with bootstrap CIs, paired comparison and ablation."),
]


def write_algorithm(out_path: Path) -> None:
    """algorithm.md: numbered pseudocode plus grouped explanations (as on sample slide 14)."""
    lines = ["# Algorithm 1: Temporal benchmark for one protocol", ""]
    lines += [f"{i}: {text}" for i, text in enumerate(ALGORITHM_LINES, start=1)]
    lines += ["", "## Grouped explanations", ""]
    lines += [f"- **{group}** — {text}" for group, text in ALGORITHM_GROUPS]
    Path(out_path).write_text("\n".join(lines) + "\n")


LATEX_ACCENTS = {r'{\"u}': "ü", r'{\"a}': "ä", r'{\"o}': "ö", r"{\'e}": "é", r"{\'E}": "É"}


def clean_latex(value: str) -> str:
    """Turn the few LaTeX constructs used in references.bib into plain text."""
    for latex, char in LATEX_ACCENTS.items():
        value = value.replace(latex, char)
    value = value.replace("\\_", "_").replace("--", "–")
    return re.sub(r"[{}]", "", value)


def parse_bib(path: Path) -> dict[str, dict]:
    """Very small BibTeX reader: {key: {field: value}} (enough for references.bib)."""
    entries = {}
    for match in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", Path(path).read_text(), re.DOTALL):
        fields = dict(re.findall(r"(\w+)\s*=\s*\{(.*?)\},?\s*\n", match.group(3) + "\n", re.DOTALL))
        fields = {k: clean_latex(v) for k, v in fields.items()}
        entries[match.group(2).strip()] = {"type": match.group(1).lower(), **fields}
    return entries


def ieee_authors(text: str) -> str:
    """'Ivan Rubachev and Nikolay Kartashev' -> 'I. Rubachev and N. Kartashev' (IEEE style)."""
    if text.startswith(("Centers", "Yandex")):  # organisations are written in full
        return text
    short = []
    for name in [n.strip() for n in text.split(" and ")]:
        parts = name.split()
        short.append(" ".join(p[0] + "." for p in parts[:-1]) + " " + parts[-1])
    if len(short) <= 2:
        return " and ".join(short)
    return ", ".join(short[:-1]) + ", and " + short[-1]


def ieee_reference(entry: dict) -> str:
    """One reference in IEEE style."""
    text = f"{ieee_authors(entry['author'])}, “{entry['title']},” "
    if "journal" in entry:
        text += (f"*{entry['journal']}*, vol. {entry['volume']}, no. {entry['number']}, "
                 f"pp. {entry['pages']}, {entry['year']}.")
    elif "booktitle" in entry:
        text += f"in *{entry['booktitle']}*, {entry['year']}"
        text += f", doi: {entry['doi']}." if "doi" in entry else "."
    elif "howpublished" in entry:
        url = entry["howpublished"].replace("\\url", "")
        text += f"{url}. {entry.get('note', '')}."
    else:
        text += f"{entry.get('note', 'preprint')}, {entry['year']}."
    if "note" in entry and "arXiv" in entry["note"] and "booktitle" in entry:
        text += f" {entry['note']}."
    return text.replace("..", ".")


def write_references(bib_path: Path, order: list[str], out_path: Path) -> None:
    """references.md in IEEE style, numbered in order of first citation in the deck."""
    entries = parse_bib(bib_path)
    lines = ["# References (IEEE)", "", "Same entries as report/references.bib.", ""]
    lines += [f"[{i}] {ieee_reference(entries[key])}" for i, key in enumerate(order, start=1)]
    Path(out_path).write_text("\n".join(lines) + "\n")


def theme_summary(pptx_path: Path) -> dict:
    """Theme colours, fonts and slide size read directly from the sample .pptx file."""
    with zipfile.ZipFile(pptx_path) as archive:
        theme = archive.read("ppt/theme/theme1.xml").decode("utf-8")
    colors = dict(re.findall(r"<a:(dk1|lt1|dk2|lt2|accent\d|hlink|folHlink)>\s*<a:\w+Clr "
                             r"(?:val|lastClr)=\"([0-9A-Fa-f]{6})\"", theme))
    fonts = dict(re.findall(r"<a:(majorFont|minorFont)>\s*<a:latin typeface=\"([^\"]*)\"", theme))
    deck = Presentation(pptx_path)
    return {"colors": colors, "fonts": fonts, "width_in": deck.slide_width / 914400,
            "height_in": deck.slide_height / 914400, "n_slides": len(deck.slides)}


def render_sample_slides(pptx_path: Path, out_dir: Path) -> str:
    """Render every sample slide to sample-NN.png with LibreOffice + pdftoppm."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (shutil.which("soffice") and shutil.which("pdftoppm")):
        return "LibreOffice (soffice) or pdftoppm not available: sample slides were not rendered."
    subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(out_dir),
                    str(pptx_path)], check=True, capture_output=True)
    pdf = out_dir / (Path(pptx_path).stem + ".pdf")
    subprocess.run(["pdftoppm", "-png", "-r", "110", str(pdf), str(out_dir / "sample")],
                   check=True, capture_output=True)
    pdf.unlink()
    for path in out_dir.glob("sample-*.png"):  # pdftoppm pads to 2 digits for 27 pages
        number = int(path.stem.split("-")[1])
        path.rename(out_dir / f"sample-{number:02d}.png")
    return f"Rendered {len(list(out_dir.glob('sample-*.png')))} slides with LibreOffice + pdftoppm."


def first_run(shape):
    """Font size (pt), bold flag and colour of the first text run of a shape."""
    run = next(r for p in shape.text_frame.paragraphs for r in p.runs)
    color = str(run.font.color.rgb) if run.font.color and run.font.color.type else "inherited"
    return run.font.size.pt if run.font.size else None, run.font.bold, color


def measured_styles(pptx_path: Path) -> dict:
    """Section label, title, body, caption and table styles measured on sample slides."""
    deck = Presentation(pptx_path)
    s3, s9 = deck.slides[2], deck.slides[8]
    by_name = {sh.name: sh for sh in s3.shapes}
    table = next(sh for sh in s9.shapes if sh.has_table).table
    header_fill = str(table.cell(0, 0).fill.fore_color.rgb)
    band_fill = str(table.cell(2, 0).fill.fore_color.rgb)
    caption = next(sh for sh in s9.shapes if sh.has_text_frame
                   and sh.text_frame.text.startswith("Table"))
    return {"label": first_run(by_name["Text 0"]), "title": first_run(by_name["Text 1"]),
            "footer": next(sh.text_frame.text for sh in s3.shapes if "Footer" in sh.name),
            "header_fill": header_fill, "band_fill": band_fill,
            "caption": first_run(caption) + (caption.text_frame.paragraphs[0].runs[0].font.name,)}


def write_style_guide(pptx_path: Path, out_path: Path, render_note: str) -> None:
    """style_guide.md: visual format of the sample deck (theme read from the .pptx)."""
    theme, m = theme_summary(pptx_path), measured_styles(pptx_path)
    colors = "\n".join(f"| {k} | #{v.upper()} |" for k, v in theme["colors"].items())
    text = f"""# Style guide (extracted from the sample deck)

Source: `reference/CSCI447_Final_Project_Sample_Presentation.pptx` ({theme['n_slides']} slides).
Rendering: {render_note}

## Slide size
{theme['width_in']:.2f} × {theme['height_in']:.2f} in (16:9). Margins: content starts 0.60 in from the left and spans 12.10 in.

## Theme colours (ppt/theme/theme1.xml)
| Role | Hex |
|---|---|
{colors}

Roles used on the slides: dk2 #14213D = titles, big numbers, table header fill; accent1 #24375F =
numbered circles (first half); accent2 #B8913A = section labels, numbered circles (second half),
big-number accent; accent4 #B4432E = warnings/limitations circle; lt2 #F3F5F9 = cards and table
banding; #F3E9D2 = takeaway and callout boxes; body text #1F2937; secondary text #5B6472;
grid/borders #D5DAE3. Closing slide ("DARK" layout) background #14213D with white title.

## Fonts
- Headings: **{theme['fonts'].get('majorFont')}** (titles {m['title'][0]:.0f} pt bold, colour #{m['title'][2]}; card titles 19–20 pt; big numbers 30–40 pt).
- Body: **{theme['fonts'].get('minorFont')}** (body 15–16 pt; secondary 13–14 pt; tables 15 pt).
- Equations: Cambria Math 18 pt; equation numbers "(n)" 15 pt in #5B6472 at the right edge.
- Captions: {m['caption'][3]} {m['caption'][0]:.0f} pt, "Table N:" placed **above** tables, "Fig. N:" placed **below** figures.

## Recurring elements
- **Section label** ("SECTION N · NAME", all caps): {m['label'][0]:.0f} pt bold, colour #{m['label'][2]}, at (0.60 in, 0.30 in).
- **Title**: {theme['fonts'].get('majorFont')} {m['title'][0]:.0f} pt, #{m['title'][2]}, at (0.60 in, 0.62 in).
- **Footer**: {len(m['footer'].split('|'))} parts separated by "   |   " — course name, "Project ID: <ID>", project title (our footer text is given on slide 1 of `slides.md`); bottom left (0.60 in, 7.05 in; 11 pt). **Slide number** bottom right (12.20 in, 7.03 in).
- **Tables**: header row fill #{m['header_fill']} with white bold 15 pt text; banded rows #{m['band_fill']} / white; centred numbers; best value per column in bold.
- **Cards**: rounded rectangles #F3F5F9 with a numbered circle (0.50 in, white bold number) and a Cambria 19–20 pt card title.
- **Big-number callouts**: rounded #F3F5F9 boxes 3.85 × 1.0–1.5 in; value in Cambria 30–34 pt bold (#14213D, or #B4432E for a cost/negative); label 14 pt #5B6472 below.
- **Takeaway box**: full-width rounded #F3E9D2 box (12.10 × 1.10 in) at the bottom; "Takeaway:" in bold 15–16 pt.
- **Roadmap**: two columns of numbered circles (navy 1–5, gold 6–9) and a #F3E9D2 presenter callout.
- **Algorithm slide**: pseudocode card on the left (Cambria 12 pt, line numbers in #5B6472), grouped explanation cards on the right alternating #F3E9D2 / #F3F5F9.

## Model colours used in our figures
Logistic Regression #1F5FA8, Random Forest #A8822C, XGBoost #109A92, LightGBM #B4432E, MLP #6B4E9B
(deck hues adjusted for colour-blind separation); cycles from light to dark navy (#9FB0CC, #4F6A99, #14213D).
"""
    Path(out_path).write_text(text)


def write_placeholders(slides_path: Path, render_note: str, out_path: Path) -> None:
    """PLACEHOLDERS.md: everything the students still have to fill in or check."""
    review, todo, current = [], [], ""
    for line in Path(slides_path).read_text().splitlines():
        if line.startswith("## Slide"):
            current = line[3:]
        if not current:  # skip the file header before the first slide
            continue
        if "[REVIEW]" in line:
            review.append(f"- {current}: {line.strip()[:160]}")
        if "TODO" in line:
            todo.append(f"- {current}: {line.strip()[:160]}")
    text = "\n".join([
        "# Placeholders and items to check", "",
        "## Names and IDs (keep exactly as written until replaced)",
        ("- `STUDENT_NAME_1`, `STUDENT_ID_1`, `STUDENT_NAME_2`, `STUDENT_ID_2` (slides 1 and 28, "
        "report, notebook title cell)."),
        ("- Notebook file name `CSCI447P034.ipynb`: the sample deck's notes say the notebook must "
        "be named `<ProjectID>.ipynb` (i.e. possibly `P034.ipynb`). Check the course rule."), "",
        "## TO VERIFY",
        "- See `report/README.md` → 'To verify' (venues of arXiv papers, LightGBM page numbers).",
        "- The project timeline (slide 6) is a plan; replace the week numbers with your real schedule.",
        "- Course IEEE template was not provided; the report uses the standard IEEEtran class.",
        f"- Sample slide rendering: {render_note}", "",
        "## TODO slides", *(todo or ["- None: every slide has real content."]), "",
        "## [REVIEW] passages (rewrite in your own words)", *review, "",
    ])
    Path(out_path).write_text(text + "\n")


def write_pack_readme(out_dir: Path, render_note: str) -> None:
    """README.md of the pack: contents, provenance of numbers, how to regenerate."""
    text = f"""# Presentation pack — Clinical TabReD Lite (P034)

This folder is self-contained: it holds everything needed to build the final presentation in the
format of the course sample deck (`reference/`).

## Contents
| Path | What it is |
|---|---|
| `slides.md` | Slide-by-slide content (28 slides) in the required block format |
| `style_guide.md` | Colours, fonts, sizes and recurring elements extracted from the sample deck |
| `key_numbers.json` | Every headline value with its source file, column and row filter |
| `PLACEHOLDERS.md` | Names/IDs, items to verify, `[REVIEW]` passages |
| `equations.md`, `algorithm.md`, `references.md` | Equations (1)–(7), Algorithm 1, IEEE references |
| `tables/` | `tNN_name.csv` + `tNN_name.md` for every table (numbered in slide order) |
| `charts_data/` | Raw data of every chart, so charts can be redrawn natively |
| `figures/` | Slide-ready figures (PNG 300 DPI + SVG) |
| `diagrams/` | Methodology, architecture, protocols and features diagrams (PNG + SVG + box/arrow description) |
| `reference/` | The sample `.pptx` and its slides rendered as `sample_slides/sample-NN.png` |

## How the numbers were produced
All numbers come from the CSV/JSON files in the project's `results/` folder, written by the
experiment pipeline on real NHANES data. Nothing is typed by hand: `slides.md`, tables, charts and
`key_numbers.json` are generated by code, and a test checks `key_numbers.json` against the CSVs.

Sample slide rendering: {render_note}

## Regenerate
From the repository root:
```bash
python -m clinical_tabred.run --config configs/experiment.yaml --stage presentation
```
"""
    (Path(out_dir) / "README.md").write_text(text)
