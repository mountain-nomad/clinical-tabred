# Time-based splits (TabReD construction)

Each random-k split pools all rows of sliding-window-k and re-splits them at random into the
same train / validation / test sizes (no temporal order).

| Dataset | Split | Train | Validation | Test |
|---|---|---|---|---|
| nhanes | default | 2005-2006, 2007-2008, 2009-2010, 2011-2012 | 2013-2014 | 2015-2016, 2017-2018 |
| nhanes | sliding-window-0 | 2005-2006, 2007-2008, 2009-2010 | 2011-2012 | 2013-2014 |
| nhanes | sliding-window-1 | 2007-2008, 2009-2010, 2011-2012 | 2013-2014 | 2015-2016 |
| nhanes | sliding-window-2 | 2009-2010, 2011-2012, 2013-2014 | 2015-2016 | 2017-2018 |
| brfss | default | 2017, 2018, 2019, 2020 | 2021 | 2022, 2023 |
| brfss | sliding-window-0 | 2017, 2018, 2019 | 2020 | 2021 |
| brfss | sliding-window-1 | 2018, 2019, 2020 | 2021 | 2022 |
| brfss | sliding-window-2 | 2019, 2020, 2021 | 2022 | 2023 |
| natality | default | 2016, 2017, 2018, 2019, 2020, 2021 | 2022 | 2023, 2024 |
| natality | sliding-window-0 | 2016, 2017, 2018 | 2019 | 2020 |
| natality | sliding-window-1 | 2018, 2019, 2020 | 2021 | 2022 |
| natality | sliding-window-2 | 2020, 2021, 2022 | 2023 | 2024 |
