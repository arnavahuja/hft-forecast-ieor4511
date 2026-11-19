"""Pull definitions out of the committed China L2 notebook so this analysis runs the exact same code."""
import json
from pathlib import Path
NB = Path(__file__).resolve().parent.parent / "L2_China_60s_Trade_Forecasting.ipynb"
cells = [c for c in json.loads(NB.read_text())["cells"]]
def src(i): return "".join(cells[i]["source"])
def find(marker):
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and marker in "".join(c["source"])]
    assert len(hits) == 1, (marker, hits); return src(hits[0])
def before(text, marker):
    assert marker in text, marker; return text.split(marker)[0]
