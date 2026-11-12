| approach | ref | data_type | split_type | key_finding |
|---|---|---|---|---|
| TabReD benchmark | [1] | Eight industry-grade tabular datasets | Time-based vs random train/test splits | "evaluation on time-based data splits leads to different methods ranking, compared to evaluation on random splits"; "simple MLP-like architectures and GBDT show the best results on the TabReD datasets" (abstract) |
| TableShift benchmark | [2] | 15 binary tabular classification tasks incl. BRFSS diabetes | Domain shift (Diabetes task: train on White non-Hispanic respondents, test on other groups) | "a strong relationship between shift gap (difference between ID and OOD performance) and shifts in the label distribution" (abstract) |
| Tree-based vs deep learning on tabular data | [3] | 45 tabular datasets | Benchmark splits (not time-based) | "tree-based models remain state-of-the-art on medium-sized data (~10K samples)" (abstract) |
| Detecting dataset shift (Failing Loudly) | [4] | Several datasets with simulated covariate and label shifts | Two-sample tests between source and target data | "domain-discriminating approaches tend to be helpful for characterizing shifts qualitatively" (abstract) |
