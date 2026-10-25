"""Checks a repository against CODING-RULES.md for what ruff and prettier do not cover.

Usage: python3 rules_check.py <repo root> [--fix-whitespace]

Exit code 1 when any finding is printed. Findings are "path:line: rule: detail".
"""

import ast
from dataclasses import dataclass
from pathlib import Path
import re
import sys


MAX_LINE = 100
MAX_FUNCTION_LINES = 50
MAX_NESTING = 3
MAX_PARAMS = 4
MIN_ARGV = 2
MIN_HEADING_WORDS = 3
MIN_README_LINES = 3
BLANK_RUN_LIMIT = 2
SKIP_DIRS = {
    ".git",
    "node_modules",
    "_site",
    "_site-drafts",
    "venv",
    ".venv",
    "dist",
    "build",
    "__pycache__",
    "fonts",
    "images",
    "archive",
    "data",
    "results",
    "baselines",
    "exp",
    "docs",
    "logs",
    "runs",
    "scratch_logs",
}
SKIP_FILES = {"CODING-RULES.md", "PROTOCOL.md"}
FROZEN_PREFIXES = ("PREREG",)
CODE_SUFFIXES = {".py", ".js", ".ts", ".css", ".html", ".md", ".yml", ".yaml", ".toml", ".sh"}
PROSE_SUFFIXES = {".md"}
BANNED_WORDS = [
    "powerful",
    "seamless",
    "robust",
    "cutting-edge",
    "leverage",
    "blazing fast",
    "revolutionary",
    "delve",
    "tapestry",
    "landscape",
    "pivotal",
    "testament",
    "it's worth noting",
    "it is worth noting",
]
BANNED_PATTERNS = [
    (re.compile("—"), "em dash"),
    (re.compile(r"\bnot just \w+[^.]{0,40}\bbut\b", re.I), "'not just X, but Y'"),
    (re.compile(r"\bit'?s not \w+[^.]{0,40},\s*it'?s\b", re.I), "'it's not X, it's Y'"),
    (re.compile(r"^#+ .*[\U0001F300-\U0001FAFF]", re.M), "emoji in heading"),
    (re.compile(r"\bas of\b", re.I), "'as of' dated claim"),
]
README_SECTIONS = ["how it works", "run it", "limitations"]
NARRATION_STARTS = (
    "loop over",
    "iterate",
    "call the",
    "set the",
    "return the",
    "increment",
    "check if",
)
PROVENANCE = re.compile(
    r"\b(ENG|ATH|AGE|PR|JIRA)-?\d+\b|\bpreviously\b|\bused to\b|\bchanged from\b"
)


@dataclass
class Finding:
    path: Path
    line: int
    rule: str
    detail: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.rule}: {self.detail}"


def main(argv: list[str]) -> int:
    if len(argv) < MIN_ARGV:
        print(__doc__)  # noqa: T201
        return 2
    root = Path(argv[1]).resolve()
    findings: list[Finding] = []
    for path in _source_files(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        findings += check_whitespace(path, text)
        if path.suffix == ".py":
            findings += check_python(path, text)
        if path.suffix in PROSE_SUFFIXES:
            findings += check_prose(path, text)
        if path.name.lower() == "readme.md":
            findings += check_readme(path, text)
    for finding in sorted(findings, key=lambda f: (str(f.path), f.line)):
        print(finding)  # noqa: T201
    print(f"{len(findings)} findings")  # noqa: T201
    return 1 if findings else 0


def check_whitespace(path: Path, text: str) -> list[Finding]:
    findings = []
    lines = text.split("\n")
    for number, line in enumerate(lines, start=1):
        if "\t" in line and path.name != "Makefile":
            findings.append(Finding(path, number, "tabs", "tab character"))
        if line.rstrip() != line and path.suffix != ".md":
            findings.append(Finding(path, number, "trailing-whitespace", ""))
        if len(line) > MAX_LINE and path.suffix in {".py", ".js", ".ts"} and "http" not in line:
            findings.append(Finding(path, number, "line-length", f"{len(line)} > {MAX_LINE}"))
    if text and not text.endswith("\n"):
        findings.append(
            Finding(path, len(lines), "final-newline", "file does not end with a newline")
        )
    if text.endswith("\n\n"):
        findings.append(
            Finding(path, len(lines), "final-newline", "more than one newline at end of file")
        )
    return findings


def check_python(path: Path, text: str) -> list[Finding]:
    findings = []
    try:
        tree = ast.parse(text)
    except SyntaxError as error:
        return [Finding(path, error.lineno or 0, "syntax", str(error.msg))]
    lines = text.split("\n")
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            findings += _check_function(path, node, lines)
    findings += _check_comments(path, lines)
    return findings


def _check_function(path: Path, node: ast.FunctionDef, lines: list[str]) -> list[Finding]:
    findings = []
    length = (node.end_lineno or node.lineno) - node.lineno + 1
    if length > MAX_FUNCTION_LINES:
        findings.append(
            Finding(path, node.lineno, "function-length", f"{node.name} is {length} lines")
        )
    params = [a for a in node.args.args + node.args.kwonlyargs if a.arg not in {"self", "cls"}]
    if len(params) > MAX_PARAMS:
        findings.append(
            Finding(path, node.lineno, "parameters", f"{node.name} takes {len(params)}")
        )
    depth = _max_nesting(node)
    if depth > MAX_NESTING:
        findings.append(Finding(path, node.lineno, "nesting", f"{node.name} nests {depth} deep"))
    if node.returns is None and node.name != "__init__":
        findings.append(
            Finding(path, node.lineno, "return-type", f"{node.name} has no return annotation")
        )
    for arg in params:
        if arg.annotation is None:
            findings.append(
                Finding(path, node.lineno, "type-hint", f"{node.name}: {arg.arg} untyped")
            )
    body_lines = lines[node.lineno : node.end_lineno or node.lineno]
    blank_run = 0
    for offset, line in enumerate(body_lines, start=node.lineno + 1):
        blank_run = blank_run + 1 if not line.strip() else 0
        if blank_run == BLANK_RUN_LIMIT:
            findings.append(
                Finding(path, offset, "blank-lines", "two blank lines inside a function")
            )
    first_body = lines[node.body[0].lineno - 1] if node.body else ""
    if node.body and not first_body.strip() and node.body[0].lineno > node.lineno + 1:
        findings.append(Finding(path, node.lineno + 1, "blank-lines", "blank line right after def"))
    return findings


def _max_nesting(node: ast.AST, depth: int = 0) -> int:
    nesting_types = (ast.If, ast.For, ast.While, ast.With, ast.Try, ast.AsyncFor, ast.AsyncWith)
    deepest = depth
    for child in ast.iter_child_nodes(node):
        child_depth = depth + 1 if isinstance(child, nesting_types) else depth
        deepest = max(deepest, _max_nesting(child, child_depth))
    return deepest


def _check_comments(path: Path, lines: list[str]) -> list[Finding]:
    findings = []
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped.startswith("#") or stripped.startswith("#!"):
            continue
        body = stripped.lstrip("#").strip()
        lowered = body.lower()
        if lowered.startswith(NARRATION_STARTS):
            findings.append(Finding(path, number, "comment-narration", body[:60]))
        if PROVENANCE.search(body):
            findings.append(Finding(path, number, "comment-provenance", body[:60]))
        if (
            re.match(r"^(todo|fixme)\b", lowered)
            and " once " not in lowered
            and " when " not in lowered
        ):
            findings.append(Finding(path, number, "todo-without-condition", body[:60]))
        if _looks_like_code(body):
            findings.append(Finding(path, number, "commented-out-code", body[:60]))
    return findings


def _looks_like_code(body: str) -> bool:
    try:
        ast.parse(body)
    except SyntaxError:
        return False
    return bool(re.search(r"[=()\[\]]|\breturn\b|\bimport\b", body)) and not body.endswith(".")


def check_prose(path: Path, text: str) -> list[Finding]:
    findings = []
    for number, line in enumerate(text.split("\n"), start=1):
        lowered = _strip_literal_spans(line).lower()
        for word in BANNED_WORDS:
            if re.search(rf"\b{re.escape(word)}\b", lowered):
                findings.append(Finding(path, number, "banned-word", word))
        for pattern, name in BANNED_PATTERNS:
            if pattern.search(line):
                findings.append(Finding(path, number, "banned-pattern", name))
        if re.match(r"^#+ ", line) and _is_title_case(line):
            findings.append(Finding(path, number, "heading-case", "use sentence case"))
    return findings


def _strip_literal_spans(line: str) -> str:
    """Drops code spans and single-asterisk italics, which quote names rather than use words."""
    without_code = re.sub(r"`[^`]*`", "", line)
    return re.sub(r"(?<!\*)\*[^*\n]+\*(?!\*)", "", without_code)


def _is_title_case(heading: str) -> bool:
    words = re.sub(r"^#+ ", "", heading).split()
    capitalised = [w for w in words[1:] if w[:1].isupper() and not w.isupper()]
    return len(words) >= MIN_HEADING_WORDS and len(capitalised) >= len(words[1:]) - 1


def check_readme(path: Path, text: str) -> list[Finding]:
    findings = []
    headings = [h.lower() for h in re.findall(r"^#+ (.+)$", text, re.M)]
    for wanted in README_SECTIONS:
        if not any(wanted in h for h in headings):
            findings.append(Finding(path, 1, "readme-structure", f"no '{wanted}' section"))
    body = text.strip().split("\n")
    if len(body) >= MIN_README_LINES and not body[0].startswith("#"):
        findings.append(Finding(path, 1, "readme-structure", "first line should be the title"))
    if re.search(r"\bwe\b", text) and "collaborat" not in text.lower():
        findings.append(Finding(path, 1, "readme-voice", "'we' in a solo project"))
    return findings


def _source_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*"):
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.name in SKIP_FILES or path.name.startswith(FROZEN_PREFIXES):
            continue
        if path.is_file() and path.suffix in CODE_SUFFIXES:
            files.append(path)
    return files


if __name__ == "__main__":
    sys.exit(main(sys.argv))
