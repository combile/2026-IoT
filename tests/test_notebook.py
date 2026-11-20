import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "colab_run.ipynb"


def test_notebook_is_valid_json_and_code_compiles():
    nb = json.loads(NB.read_text())                     # 충돌 해결 중 줄바꿈이 깨지면 코랩이 노트북을 못 연다
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            src = "".join(l for l in cell["source"] if not l.lstrip().startswith(("!", "%")))
            compile(src, "cell", "exec")
