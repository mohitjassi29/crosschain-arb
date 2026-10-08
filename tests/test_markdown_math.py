"""GitHub renders math after markdown processing, which breaks some LaTeX.

* A backslash before ASCII punctuation (comma, brace, semicolon, bang, bar, ...) is treated as a
  markdown escape: the backslash is dropped, so a thin-space command shows as a stray comma and
  an escaped brace loses its brace.
* GitHub's renderer rejects macros such as operatorname.
"""

import re
from pathlib import Path

import nbformat
import pytest

ROOT = Path(__file__).resolve().parents[1]
BANNED = (
    "\\operatorname",
    "\\DeclareMathOperator",
    "\\newcommand",
    "\\def",
    "\\require",
    "\\href",
    "\\class",
    "\\style",
)
ESCAPED_PUNCT = re.compile(r"\\[!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]")


def math_segments(text: str):
    blocks = re.findall(r"\$\$(.+?)\$\$", text, flags=re.S)
    rest = re.sub(r"\$\$.+?\$\$", " ", text, flags=re.S)
    inline = re.findall(r"(?<!\$)\$(?!\$)([^$\n]+?)\$(?!\$)", rest)
    return blocks + inline


def markdown_sources():
    yield "README.md", (ROOT / "README.md").read_text()
    for f in sorted((ROOT / "notebooks").glob("*.ipynb")):
        for i, c in enumerate(nbformat.read(f, as_version=4).cells):
            if c.cell_type == "markdown":
                yield f"{f.name}[cell {i}]", c.source


@pytest.mark.parametrize("name,text", list(markdown_sources()))
def test_math_survives_github_markdown(name, text):
    for seg in math_segments(text):
        bad = ESCAPED_PUNCT.findall(seg)
        assert not bad, f"{name}: backslash before punctuation {bad} in math: {seg[:80]!r}"
        for macro in BANNED:
            assert macro not in seg, f"{name}: {macro} is not allowed by GitHub math: {seg[:80]!r}"


def test_readme_has_math_to_check():
    assert len(math_segments((ROOT / "README.md").read_text())) > 20
