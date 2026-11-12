## 1. Project introduction

Machine learning models for healthcare are usually trained on data collected in the past and then used on
patients seen later. If the population, measurement practice or disease prevalence changes over time, a
model can perform worse than a standard random train/test split suggests. The TabReD benchmark
(Rubachev et al., 2024) showed that, for industrial tabular datasets, **time-based splits** can change
which methods look best compared with random splits.

This project is a small, educational version of that idea on public health-survey data. We predict
whether an adult reports having been told by a doctor that they have diabetes, using demographic,
socioeconomic and body-measurement features, and compare:

* **Protocol A (chronological):** train on 2011-2012, validate on 2013-2014, test on 2015-2016.
* **Protocol A2 (in-period reference):** the same model, tested both on a held-out 20% of 2011-2012 and on 2015-2016.
* **Protocol B (random):** all cycles pooled and split randomly 70/15/15 (5 seeds).
* **5-fold stratified cross-validation** on the development data of A and B.

**Research questions**
1. How does performance differ between random and chronological splits?
2. Which models generalise better to a newer survey cohort?
3. Which feature distributions and target prevalences change between cycles?
4. Can these observed changes help explain performance differences?

This is a prototype for learning, **not** a validated clinical prediction tool.
