"""key_numbers.json in for_presentation/ must match the result files it cites."""
import json

from conftest import REPO
from make_presentation_pack import key_numbers


def test_key_numbers_match_results():
    saved = json.loads((REPO / "for_presentation" / "key_numbers.json").read_text())
    assert saved == json.loads(json.dumps(key_numbers()))


def test_slide_visuals_exist():
    pack = REPO / "for_presentation"
    for line in (pack / "slides.md").read_text().splitlines():
        if line.startswith("Visuals:") and "none" not in line:
            for item in line.removeprefix("Visuals:").split("|"):
                path = item.strip().split(" ")[0]
                assert (pack / path).exists() or (pack / path.replace(".svg", ".png")).exists(), path
