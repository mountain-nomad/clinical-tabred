## 3. Data loading

The CDC website sometimes returns an HTML error page with HTTP status 200 instead of the data file.
Therefore every file is **validated**: it must not start with `<`, it must start with the SAS transport
header, and pandas must be able to read it. Valid files are cached in `./nhanes_data/`, and a manifest
records the URL, HTTP status, size and SHA-256 hash of every file. If a download fails, the error message
lists the exact URLs that were tried.

<!-- source: download.XPT_MAGIC download.build_urls download.validate_xpt download.sha256_of download.fetch_url download.download_component download.download_all -->

```python
manifest = download_all(CONFIG, RAW_DIR)
manifest[["component", "cycle", "url", "http_status", "bytes", "validation"]]
```

Reading an XPT file: `pandas.read_sas` returns all numbers as floats and NHANES stores some zeros as tiny
values such as 5.4e-79, which we round to exactly 0. We also check that the respondent ID is unique.

<!-- source: prepare.read_xpt -->

```python
for path in sorted(RAW_DIR.glob("*.xpt")):
    table = read_xpt(path)
    print(f"{path.name:12s} rows={len(table):6,d} columns={table.shape[1]}")
```
