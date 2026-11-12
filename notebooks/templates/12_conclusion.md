## 12. Limitations and conclusion

### Limitations
* **Survey, not hospital data:** NHANES is a cross-sectional population survey, not longitudinal
  electronic health records; each cycle contains different people.
* **Unweighted models:** NHANES uses complex multistage sampling with survey weights. Our models are
  trained without weights, so the results describe the *sample*, not the US population (weighted
  prevalence is reported only as a descriptive check).
* **Self-reported target:** the label is self-reported diagnosed diabetes, so undiagnosed diabetes counts
  as negative.
* **Reverse causation:** body measurements are taken after a possible diagnosis, so associations may
  partly reflect treatment or lifestyle changes after diagnosis.
* **Small scale:** three cycles, nine features and small hyperparameter grids; this is a prototype, not
  a full reproduction of TabReD.
* **Random vs temporal test sets differ** in who is in them, so their difference is not purely a time effect.

### Conclusion
The key findings and their rule-based interpretation in Section 11 summarise the measured results.
Students: write the final interpretation here in your own words, using only the numbers shown above.
