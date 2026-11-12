# Clinical TabReD Lite
## Evaluating Tabular Machine Learning Models under Temporal Distribution Shift in Healthcare Data

**Course:** CSCI 447 Machine Learning: Theory and Practice, Fall 2026  
**Project ID:** P034  
**Students:** STUDENT_NAME_1 (STUDENT_ID_1), STUDENT_NAME_2 (STUDENT_ID_2)  
**Instructor:** Dr. Hari Mohan Rai, Assistant Professor, Department of Computer Science, SCAI, Nazarbayev University

This notebook is **self-contained**: it downloads the public NHANES data, prepares it, trains five models,
evaluates them under three protocols and analyses temporal drift. It needs only internet access and a
standard Python 3.10+ Jupyter installation. All code below is the same code as in the project repository
(copied automatically), and every number shown is computed while the notebook runs.

**Contents:** 1 Introduction · 2 Dataset · 3 Data loading · 4 Preprocessing · 5 Exploratory analysis ·
6 Train/test splitting · 7 Cross-validation · 8 Model training · 9 Model evaluation · 10 Temporal drift ·
11 Results and comparison · 12 Limitations and conclusion

### Setup
The first cell checks that the required packages are installed and installs only the missing ones,
pinned to the versions this notebook was tested with.

<!-- setup -->

<!-- imports -->

All settings live in one `CONFIG` dictionary. Paths are relative to the notebook:
raw data is cached in `./nhanes_data/`, and all tables and figures are written to `./notebook_outputs/`.

<!-- config -->
