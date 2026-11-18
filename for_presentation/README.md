# Presentation pack — Clinical TabReD (P034)

Everything needed to build the final presentation in the format of the course sample deck
(`reference/`). Content and numbers are from the three-dataset benchmark (NHANES, BRFSS, natality);
the full write-up is `../docs/RESULTS.md`.

| Path | What it is |
|---|---|
| `GUIDE_RU.md` | Full guide in Russian for the presenter (pipeline, models, metrics, results, FAQ, glossary) |
| `slides.md` | Slide-by-slide content (29 slides): layout, section label, content, visuals, captions, big numbers, speaker notes |
| `style_guide.md` | Colours, fonts, sizes and recurring elements of the sample deck |
| `key_numbers.json` | Headline numbers with the result file and filter each comes from |
| `PLACEHOLDERS.md` | Items the presenters must check |
| `equations.md`, `algorithm.md`, `references.md` | Equations (1)–(8), Algorithm 1, IEEE references |
| `tables/` | `tNN_*.csv` + `.md` for every table |
| `charts_data/` | Raw data of every chart, to redraw charts natively |
| `figures/` | Result figures (PNG 300 DPI + SVG), captions in `figures/captions.md` |
| `diagrams/` | Methodology, model pipeline and split diagrams (PNG + SVG + box/arrow description `.md`) |
| `reference/` | The course sample `.pptx` and its slides as `sample_slides/sample-NN.png` |

## Regenerate
`figures/`, `diagrams/`, `tables/`, `charts_data/` and `key_numbers.json` are generated from the
saved results (the other files are written by hand):
```bash
python scripts/summarize_benchmarks.py && python scripts/make_figures.py && python scripts/make_diagrams.py
python scripts/make_presentation_pack.py
```
