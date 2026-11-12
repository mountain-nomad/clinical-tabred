# Features and preprocessing

## Boxes
1. **Demographic RIDAGEYR, RIAGENDR, RIDRETH3** (position x=0, y=2.8)
2. **Socioeconomic DMDEDUC2, INDFMPIR** (position x=0, y=1.4)
3. **Anthropometric BMXBMI, BMXWAIST, BMXHT, BMXWT** (position x=0, y=0.0)
4. **Numeric (6) median imputation → standard scaling** (position x=3.6, y=2.1)
5. **Categorical (3) mode imputation → one-hot encoding** (position x=3.6, y=0.7)
6. **Model input (fit on training split only)** (position x=7.2, y=1.4)

## Arrows
- Demographic RIDAGEYR, RIAGENDR, RIDRETH3 → Numeric (6) median imputation → standard scaling
- Demographic RIDAGEYR, RIAGENDR, RIDRETH3 → Categorical (3) mode imputation → one-hot encoding
- Socioeconomic DMDEDUC2, INDFMPIR → Numeric (6) median imputation → standard scaling
- Socioeconomic DMDEDUC2, INDFMPIR → Categorical (3) mode imputation → one-hot encoding
- Anthropometric BMXBMI, BMXWAIST, BMXHT, BMXWT → Numeric (6) median imputation → standard scaling
- Numeric (6) median imputation → standard scaling → Model input (fit on training split only)
- Categorical (3) mode imputation → one-hot encoding → Model input (fit on training split only)
