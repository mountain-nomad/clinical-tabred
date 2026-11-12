# Report (IEEE conference format)

- `main.tex` – report source (IEEEtran conference class).
- `tables/*.tex` – **generated** from `../results/` by `python -m clinical_tabred.run --stage report`.
  `tables/numbers.tex` defines `\kn{<id>}` macros, so every number in the text comes from the results.
- `literature/*_curated.csv` – literature facts and values copied exactly from the sources listed below.
- `timeline.csv` – the **planned** project timeline (a plan, not a result). Adjust the weeks to your real schedule.
- Figures are included from `../figures/` (PDF versions).

## Build
```bash
python -m clinical_tabred.run --config configs/experiment.yaml --stage report   # tables + compile
```
The stage uses `latexmk -pdf` if available, otherwise `tectonic` (used for this build: LaTeX/latexmk
were not installed on the development machine, so `tectonic` was installed with Homebrew).

## Template
The course IEEE template (`report/template/`) was **not provided**, so the standard `IEEEtran`
conference class is used. **Students must check the report against the official course template
and move the content into it if required.**

## Student writing
Sections marked with a red `STUDENT TO WRITE` box (`\STUDENT{...}` in `main.tex`) are motivation,
interpretation and discussion. They must be written by the students in their own words. The
generated text only describes methods and reports numbers.

## References
All entries in `references.bib` were opened and checked on 2026-11-12:
TabReD (arXiv:2406.19380) and its GitHub repository, TableShift (arXiv:2312.07577, incl. HTML Table 6
and Appendix B.5), Grinsztajn et al. (arXiv:2207.08815), Rabanser et al. (arXiv:1810.11953),
scikit-learn (JMLR 12), XGBoost (arXiv:1603.02754 / KDD'16 DOI), LightGBM (NeurIPS 2017 proceedings page),
the NHANES website and the NHANES codebooks for DEMO/BMX/DIQ (cycles G, H, I).

### To verify
- TabReD venue: only the arXiv preprint was confirmed; check whether a peer-reviewed version should be cited.
- Grinsztajn et al. venue: only arXiv was confirmed (a NeurIPS Datasets & Benchmarks version may exist).
- LightGBM page numbers were not shown on the proceedings page.
- Any external statistic on diabetes prevalence you add to the introduction must be cited from an opened source.
