"""Extract value codes and labels for every variable from the CDC BRFSS LLCP codebooks 2017-2023.

Input : data/raw/brfss/codebooks/  (saved by hand from cdc.gov, which blocks scripted downloads)
        codebook17/18/20/21_llcp-v2-508.pdf   (PDF codebooks; converted with `pdftotext -layout`)
        codebook19_llcp.html                  (HTML codebook)
        codebook22/23_llcp-v2-508.zip         (zipped HTML codebooks)
Output: configs/brfss_codebook.json  {year: {variable: [[value, label], ...]}}

The build script (scripts/build_brfss_table.py) uses the labels to find the codes meaning
"Don't know / Not sure", "Refused", "Not asked or Missing" (-> missing) and "None" / "Never"
(-> 0 for count questions), exactly as configs/nhanes_missing_codes.json is used for NHANES.
"""
import html
import json
import re
import subprocess
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CB = ROOT / "data" / "raw" / "brfss" / "codebooks"
OUT = ROOT / "configs" / "brfss_codebook.json"
SOURCES = {2017: "codebook17_llcp-v2-508.pdf", 2018: "codebook18_llcp-v2-508.pdf",
           2019: "codebook19_llcp.html", 2020: "codebook20_llcp-v2-508.pdf",
           2021: "codebook21_llcp-v2-508.pdf", 2022: "codebook22_llcp-v2-508.zip",
           2023: "codebook23_llcp-v2-508.zip"}
NAME = re.compile(r"SAS\s+Variable\s+Name:\s*([A-Z_0-9]+)")


def clean(s: str) -> str:
    s = html.unescape(re.sub(r"<br\s*/?>", " | ", s))
    s = re.sub(r"<[^>]+>", "", s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_html(text: str) -> dict:
    text = text.replace("&nbsp;", " ")
    out = {}
    parts = NAME.split(text)
    for name, block in zip(parts[1::2], parts[2::2]):
        body = block.split("<tbody>", 1)[-1].split("</tbody>", 1)[0]
        rows = []
        for tr in re.findall(r"<tr>(.*?)</tr>", body, flags=re.S):
            cells = [clean(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
            if len(cells) >= 2:
                rows.append([cells[0], cells[1].split(" | Notes:")[0]])
        out.setdefault(name, rows)
    return out


ROW = re.compile(r"^\s{0,12}(BLANK|HIDDEN|\.|\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s{2,}(\S.*?)(?:\s{2,}[\d,]+\s+[\d.]+\s+[\d.]+\s*)?$")


def parse_pdf_text(text: str) -> dict:
    out = {}
    parts = NAME.split(text)
    for name, block in zip(parts[1::2], parts[2::2]):
        rows, in_table = [], False
        for line in block.splitlines():
            if re.match(r"^\s*Value\s+Value Label", line):
                in_table = True
                continue
            if re.match(r"^\s*Label:", line):
                break
            if in_table:
                m = ROW.match(line)
                if m:
                    label = re.split(r"\s{2,}", m.group(2))[0]
                    rows.append([re.sub(r"\s+", " ", m.group(1)), label.strip()])
        out.setdefault(name, rows)
    return out


def main():
    codebook = {}
    for year, fname in SOURCES.items():
        path = CB / fname
        if not path.exists():
            raise SystemExit(f"Missing {path}: save the {year} LLCP codebook from cdc.gov first")
        if path.suffix == ".pdf":
            with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
                subprocess.run(["pdftotext", "-layout", str(path), tmp.name], check=True)
                parsed = parse_pdf_text(Path(tmp.name).read_text(errors="ignore"))
        elif path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                parsed = parse_html(z.read(z.namelist()[0]).decode("cp1252", errors="replace"))
        else:
            parsed = parse_html(path.read_text(encoding="utf-8", errors="ignore"))
        codebook[str(year)] = parsed
        print(f"{year}: {len(parsed)} variables, {sum(len(v) for v in parsed.values())} value rows")
    OUT.write_text(json.dumps(codebook, indent=0))
    print("Saved", OUT)


if __name__ == "__main__":
    main()
