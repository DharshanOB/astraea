"""Tests for the compiler-aware GitHub PR Review Bot (NFC proof, regressions, blast radius)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from indexer.llvm_ir_pass import resolve_clang_binary
from pr_bot.analyzer import (
    PRReviewReport,
    analyze_git_pr,
    analyze_pr_directories,
    analyze_repository,
    parse_unified_diff,
    validate_git_ref,
)
from pr_bot.cli import main as cli_main
from pr_bot.gemini_reviewer import build_review_prompt, generate_vertex_gemini_review


def test_parse_unified_diff() -> None:
    diff = """diff --git a/src/clamp.c b/src/clamp.c
--- a/src/clamp.c
+++ b/src/clamp.c
@@ -1,4 +1,3 @@
"""
    hunks = parse_unified_diff(diff)
    assert hunks == {"src/clamp.c": [(1, 3)]}


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed")
def test_pr_bot_nfc_proof_and_blast_radius(tmp_path: Path) -> None:
    base = tmp_path / "base"
    head = tmp_path / "head"
    (base / "src").mkdir(parents=True)
    (head / "src").mkdir(parents=True)

    # Base: if-else clamp + caller in main.c
    (base / "src" / "clamp.c").write_text(
        "int clamp(int x) {\n    if (x > 0) return x;\n    return 0;\n}\n",
        encoding="utf-8",
    )
    (base / "src" / "main.c").write_text(
        "int clamp(int x);\nint run(int v) { return clamp(v); }\n",
        encoding="utf-8",
    )

    # Head: refactored to ternary clamp (NFC) + main.c unchanged
    (head / "src" / "clamp.c").write_text(
        "int clamp(int x) {\n    return (x > 0) ? x : 0;\n}\n",
        encoding="utf-8",
    )
    (head / "src" / "main.c").write_text(
        "int clamp(int x);\nint run(int v) { return clamp(v); }\n",
        encoding="utf-8",
    )

    diff = """diff --git a/src/clamp.c b/src/clamp.c
--- a/src/clamp.c
+++ b/src/clamp.c
@@ -1,4 +1,3 @@
"""
    report = analyze_pr_directories(base, head, diff, base_ref="main", head_ref="feat/ternary")
    assert len(report.changed_functions) == 1
    fn = report.changed_functions[0]
    assert fn.symbol_name == "clamp"
    assert fn.nfc_verified is True
    assert fn.ir_similarity is not None and fn.ir_similarity >= 0.99
    assert len(fn.blast_radius_callers) >= 1

    md = report.to_markdown()
    assert "NFC Verified" in md
    assert "Blast Radius Graph" in md
    assert "mermaid" in md


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed")
def test_pr_bot_lowering_regression_and_inline_comments(tmp_path: Path) -> None:
    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()

    (base / "calc.c").write_text(
        "int compute(int x) {\n    return x * 4;\n}\n",
        encoding="utf-8",
    )
    # Head adds a loop + external call, preventing branchless lowering
    (head / "calc.c").write_text(
        "int ext_fn(int v);\n"
        "int compute(int x) {\n"
        "    int acc = 0;\n"
        "    for (int i = 0; i < x; i++) {\n"
        "        if (i % 2 == 0) acc += ext_fn(i);\n"
        "        else acc -= i;\n"
        "    }\n"
        "    if (acc < 0) return 0;\n"
        "    return acc;\n"
        "}\n",
        encoding="utf-8",
    )

    diff = """diff --git a/calc.c b/calc.c
--- a/calc.c
+++ b/calc.c
@@ -1,3 +1,10 @@
"""
    report = analyze_pr_directories(base, head, diff)
    categories = {f.category for f in report.findings}
    assert "COMPLEXITY_SPIKE" in categories
    assert "LOWERING_REGRESSION" in categories

    comments = report.to_github_review_comments()
    assert len(comments) >= 1
    assert comments[0]["path"] == "calc.c"
    assert comments[0]["side"] == "RIGHT"

    prompt = build_review_prompt(report, diff)
    assert "Deterministic Analysis Report" in prompt
    assert "LOWERING_REGRESSION" in prompt
    assert "Unified Git Diff" in prompt


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed")
def test_periodic_repository_audit(tmp_path: Path) -> None:
    (tmp_path / "a.c").write_text(
        "int clamp_a(int x) {\n    if (x > 0) return x;\n    return 0;\n}\n",
        encoding="utf-8",
    )
    (tmp_path / "b.c").write_text(
        "int clamp_b(int y) {\n    return (y > 0) ? y : 0;\n}\n",
        encoding="utf-8",
    )
    repo_report = analyze_repository(tmp_path, clone_threshold=0.92)
    assert repo_report.indexed_units == 2
    assert len(repo_report.clone_clusters) >= 1
    md = repo_report.to_markdown()
    assert "Periodic 4-Layer Repository Review" in md
    assert "Semantic / IR Clone Clusters" in md


def test_python_bytecode_and_javascript_indexing(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text(
        "def compute(x: int) -> int:\n"
        "    if x > 0:\n"
        "        return x * 2\n"
        "    return 0\n",
        encoding="utf-8",
    )
    (tmp_path / "helper.js").write_text(
        "class Helper {\n"
        "  greet(name) {\n"
        "    return name ? `hi ${name}` : 'hi';\n"
        "  }\n"
        "}\n"
        "function add(a, b) {\n"
        "  return a + b;\n"
        "}\n",
        encoding="utf-8",
    )
    repo_report = analyze_repository(tmp_path)
    assert repo_report.indexed_units >= 4
    assert repo_report.degraded_units == 0


def test_pr_uses_merge_base_not_branch_tip(tmp_path: Path) -> None:
    """PR analysis must diff against merge-base so unrelated main commits are not flagged."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    git("init", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")

    (repo / "mod.py").write_text(
        "def f(x: int) -> int:\n    return x + 1\n\n"
        "def g(y: int) -> int:\n    return y * 2\n",
        encoding="utf-8",
    )
    git("add", "mod.py")
    git("commit", "-m", "initial")

    # Feature branch modifies only f
    git("checkout", "-b", "feat")
    (repo / "mod.py").write_text(
        "def f(x: int) -> int:\n    return x + 10\n\n"
        "def g(y: int) -> int:\n    return y * 2\n",
        encoding="utf-8",
    )
    git("commit", "-am", "feat modifies f")

    # Main advances independently by modifying g
    git("checkout", "main")
    (repo / "mod.py").write_text(
        "def f(x: int) -> int:\n    return x + 1\n\n"
        "def g(y: int) -> int:\n    return y * 3 + 1\n",
        encoding="utf-8",
    )
    git("commit", "-am", "main modifies g")

    report = analyze_git_pr(repo, base_ref="main", head_ref="feat")
    changed_names = {fn.symbol_name for fn in report.changed_functions}
    assert changed_names == {"f"}


def test_cli_repo_mode_and_corrupt_baseline(tmp_path: Path) -> None:
    """Verify CLI repo mode writes summary, baseline, and handles corrupt baseline JSON."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def alpha(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    out_md = tmp_path / "summary.md"
    out_comments = tmp_path / "comments.json"
    baseline = tmp_path / "baseline.json"
    baseline.write_text("{not valid json", encoding="utf-8")

    rc = cli_main(
        [
            "--mode",
            "repo",
            "--repo",
            str(repo),
            "--baseline",
            str(baseline),
            "--out-baseline",
            str(baseline),
            "--out-md",
            str(out_md),
            "--out-comments",
            str(out_comments),
        ]
    )
    assert rc == 0
    assert "Baseline Warning" in out_md.read_text(encoding="utf-8")
    assert '"hotspots"' in baseline.read_text(encoding="utf-8")


def test_pr_bot_deleted_function_blast_radius_and_inline_hunk_anchoring(tmp_path: Path) -> None:
    """Deleting a referenced function reports BLAST_RADIUS and inline lines anchor to hunk."""
    base = tmp_path / "base"
    head = tmp_path / "head"
    (base / "pkg").mkdir(parents=True)
    (head / "pkg").mkdir(parents=True)

    (base / "pkg" / "legacy.py").write_text(
        "def legacy_fn(x: int) -> int:\n    return x + 1\n",
        encoding="utf-8",
    )
    (base / "pkg" / "app.py").write_text(
        "from pkg.legacy import legacy_fn\n"
        "def compute(x: int) -> int:\n"
        "    a = legacy_fn(x)\n"
        "    b = a + 1\n"
        "    if b > 0:\n"
        "        return b\n"
        "    return 0\n",
        encoding="utf-8",
    )
    # Head deletes legacy.py and increases complexity inside compute() at line 4.
    (head / "pkg" / "app.py").write_text(
        "def compute(x: int) -> int:\n"
        "    if x == 1:\n"
        "        return 1\n"
        "    if x == 2 or x == 3:\n"
        "        return 2\n"
        "    if x == 4 and x != 0:\n"
        "        return 4\n"
        "    return 0\n",
        encoding="utf-8",
    )
    # Build explicit single-hunk diff touching line 4 in head (inside compute lines 1..8)
    diff_text = (
        "diff --git a/pkg/legacy.py b/pkg/legacy.py\n"
        "--- a/pkg/legacy.py\n"
        "+++ /dev/null\n"
        "@@ -1,2 +0,0 @@\n"
        "-def legacy_fn(x: int) -> int:\n"
        "-    return x + 1\n"
        "diff --git a/pkg/app.py b/pkg/app.py\n"
        "--- a/pkg/app.py\n"
        "+++ b/pkg/app.py\n"
        "@@ -5,2 +4,2 @@\n"
    )
    report = analyze_pr_directories(base, head, diff_text=diff_text)
    blast_syms = {f.symbol_name for f in report.findings if f.category == "BLAST_RADIUS"}
    assert "legacy_fn" in blast_syms
    comp_finding = next(f for f in report.findings if f.symbol_name == "compute")
    assert comp_finding.line == 4
    comments = report.to_github_review_comments()
    comp_comment = next(c for c in comments if c["path"] == "pkg/app.py")
    assert comp_comment["line"] == 4


@pytest.mark.skipif(resolve_clang_binary() is None, reason="clang-18/clang not installed")
def test_pr_bot_nfc_false_when_head_loses_binary_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NFC verification must be False when base has machine-code blocks but head has none."""
    from indexer.binary_lifter import BinaryLifter
    from indexer.models import BinaryBasicBlock

    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()
    code = "int same_fn(int x) { return x + 1; }\n"
    (base / "f.c").write_text(code, encoding="utf-8")
    (head / "f.c").write_text(code, encoding="utf-8")

    orig_lift = BinaryLifter.lift_elf
    calls = {"n": 0}

    def toggle_lift(self: BinaryLifter, obj_path: Path) -> dict[str, list[BinaryBasicBlock]]:
        calls["n"] += 1
        if calls["n"] >= 2:
            return {}
        return orig_lift(self, obj_path)

    monkeypatch.setattr(BinaryLifter, "lift_elf", toggle_lift)
    diff_text = "diff --git a/f.c b/f.c\n--- a/f.c\n+++ b/f.c\n@@ -1,1 +1,1 @@\n"
    report = analyze_pr_directories(base, head, diff_text=diff_text)
    delta = next(d for d in report.changed_functions if d.symbol_name == "same_fn")
    assert delta.binary_blocks_before is not None and delta.binary_blocks_before > 0
    assert delta.binary_blocks_after == 0
    assert delta.nfc_verified is False


def test_remap_compile_commands_for_worktree(tmp_path: Path) -> None:
    """_remap_compile_commands rewrites repo_root paths to worktree_root."""
    import json

    from pr_bot.analyzer import _remap_compile_commands

    repo_root = tmp_path / "repo"
    wt_root = tmp_path / "wt"
    repo_root.mkdir()
    wt_root.mkdir()
    src_cc = tmp_path / "compile_commands.json"
    src_cc.write_text(
        json.dumps(
            [
                {
                    "directory": str(repo_root),
                    "file": str(repo_root / "src" / "main.c"),
                    "arguments": ["clang", f"-I{repo_root / 'include'}", "-O2"],
                }
            ]
        ),
        encoding="utf-8",
    )
    out_cc = tmp_path / "remapped.json"
    res = _remap_compile_commands(src_cc, repo_root, wt_root, out_cc)
    assert res == out_cc
    data = json.loads(out_cc.read_text(encoding="utf-8"))
    assert data[0]["directory"] == str(wt_root.resolve())
    assert data[0]["file"] == str(wt_root.resolve() / "src" / "main.c")
    assert f"-I{wt_root.resolve() / 'include'}" in data[0]["arguments"]


def test_pr_bot_skips_enclosing_class_container_and_handles_pure_deletion_hunk(
    tmp_path: Path,
) -> None:
    """Modifying a class method or deleting its first line should only report the method."""
    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()

    (base / "svc.py").write_text(
        "# header\n"
        "def top_fn(x: int) -> int:\n"
        "    y = x + 1\n"
        "    return y\n\n"
        "class Svc:\n"
        "    def a(self, x: int) -> int:\n"
        "        return x + 1\n",
        encoding="utf-8",
    )
    (head / "svc.py").write_text(
        "# header\n"
        "def top_fn(x: int) -> int:\n"
        "    return x\n\n"
        "class Svc:\n"
        "    def a(self, x: int) -> int:\n"
        "        return x + 2\n",
        encoding="utf-8",
    )
    # Pure-deletion hunk at line 1 (immediately before def top_fn on line 2) + method edit
    diff_text = (
        "diff --git a/svc.py b/svc.py\n"
        "--- a/svc.py\n"
        "+++ b/svc.py\n"
        "@@ -2,1 +1,0 @@\n"
        "@@ -8,1 +7,1 @@\n"
    )
    report = analyze_pr_directories(base, head, diff_text=diff_text)
    changed_syms = {fn.symbol_name for fn in report.changed_functions}
    assert "Svc" not in changed_syms
    assert changed_syms == {"top_fn", "a"}


def test_deleted_function_review_comment_uses_left_side(tmp_path: Path) -> None:
    """Deleted-function BLAST_RADIUS findings must set side='LEFT' on GitHub comments."""
    base = tmp_path / "base"
    head = tmp_path / "head"
    base.mkdir()
    head.mkdir()
    (base / "mod.py").write_text(
        "def old_helper(x: int) -> int:\n    return x + 1\n\n"
        "def caller(x: int) -> int:\n    return old_helper(x)\n",
        encoding="utf-8",
    )
    (head / "mod.py").write_text(
        "def caller(x: int) -> int:\n    return x\n",
        encoding="utf-8",
    )
    diff_text = (
        "diff --git a/mod.py b/mod.py\n"
        "--- a/mod.py\n"
        "+++ b/mod.py\n"
        "@@ -1,5 +1,2 @@\n"
    )
    report = analyze_pr_directories(base, head, diff_text=diff_text)
    comments = report.to_github_review_comments()
    deleted_comment = next(c for c in comments if "old_helper" in c["body"])
    assert deleted_comment["side"] == "LEFT"


def test_validate_git_ref_rejects_flag_injection() -> None:
    """validate_git_ref must reject leading dashes, '..' traversal, and control characters."""
    assert validate_git_ref("main") == "main"
    assert validate_git_ref("origin/feat-123") == "origin/feat-123"
    with pytest.raises(ValueError, match="Invalid or unsafe git ref"):
        validate_git_ref("--upload-pack=evil")
    with pytest.raises(ValueError, match="Invalid or unsafe git ref"):
        validate_git_ref("-oProxyCommand=evil")
    with pytest.raises(ValueError, match="Invalid or unsafe git ref"):
        validate_git_ref("HEAD..origin/main")
    with pytest.raises(ValueError, match="Invalid or unsafe git ref"):
        validate_git_ref("main\n--flag")


def test_cli_repo_mode_omits_comments_file_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running astraea-review --mode repo without --out-comments must not write comments JSON."""
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def alpha(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    out_md = tmp_path / "repo_summary.md"

    rc = cli_main(["--mode", "repo", "--repo", str(repo), "--out-md", str(out_md)])
    assert rc == 0
    assert out_md.exists()
    assert not (tmp_path / "pr_review_comments.json").exists()


def test_generate_vertex_gemini_review_auth_modes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify generate_vertex_gemini_review configures genai.Client for all auth modes."""
    from google import genai

    calls: list[dict[str, Any]] = []

    class _FakeModels:
        def generate_content(self, **kwargs: Any) -> Any:
            return type("Resp", (), {"text": "  Reviewed OK  "})()

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            calls.append(kwargs)
            self.models = _FakeModels()

    monkeypatch.setattr(genai, "Client", _FakeClient)
    for var in (
        "VERTEX_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_GENAI_USE_VERTEXAI",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GCP_PROJECT_ID",
    ):
        monkeypatch.delenv(var, raising=False)

    report = PRReviewReport(base_ref="main", head_ref="feat")

    # 1. Standard GEMINI_API_KEY mode
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-1")
    out1 = generate_vertex_gemini_review(report, diff_text="+x")
    assert out1 == "Reviewed OK"
    assert calls[-1]["api_key"] == "test-key-1"
    assert "vertexai" not in calls[-1]

    # 2. Vertex AI + API key mode
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    out2 = generate_vertex_gemini_review(report, diff_text="+x")
    assert out2 == "Reviewed OK"
    assert calls[-1]["api_key"] == "test-key-1"
    assert calls[-1]["vertexai"] is True

    # 3. GCP project + location mode
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI")
    out3 = generate_vertex_gemini_review(
        report, diff_text="+x", project="my-proj", location="us-central1"
    )
    assert out3 == "Reviewed OK"
    assert calls[-1]["vertexai"] is True
    assert calls[-1]["project"] == "my-proj"
    assert calls[-1]["location"] == "us-central1"

    # 4. Antigravity session token / GOOGLE_APPLICATION_CREDENTIALS mode
    cred_file = tmp_path / "acp_token.json"
    cred_file.write_text(json.dumps({"project_id": "aicode-consumers"}), encoding="utf-8")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(cred_file))
    out4 = generate_vertex_gemini_review(report, diff_text="+x")
    assert out4 == "Reviewed OK"
    assert calls[-1]["vertexai"] is True
    assert calls[-1]["project"] == "aicode-consumers"
    monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS")

    # 5. Missing credentials raises ValueError
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        generate_vertex_gemini_review(report, diff_text="+x")


def test_cli_pr_mode_end_to_end(tmp_path: Path) -> None:
    """Verify CLI --mode pr runs end-to-end against a 2-commit git repository."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *args: subprocess.run(  # noqa: E731
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    git("init", "-b", "main")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test User")

    (repo / "calc.py").write_text(
        "def scale(x: int) -> int:\n    return x * 2\n",
        encoding="utf-8",
    )
    git("add", "calc.py")
    git("commit", "-m", "initial")

    (repo / "calc.py").write_text(
        "def scale(x: int) -> int:\n    return x * 4\n",
        encoding="utf-8",
    )
    git("commit", "-am", "update scale")

    out_md = tmp_path / "pr_summary.md"
    out_comments = tmp_path / "pr_comments.json"
    rc = cli_main(
        [
            "--mode",
            "pr",
            "--repo",
            str(repo),
            "--base",
            "HEAD~1",
            "--head",
            "HEAD",
            "--out-md",
            str(out_md),
            "--out-comments",
            str(out_comments),
        ]
    )
    assert rc == 0
    assert out_md.exists()
    assert "scale" in out_md.read_text(encoding="utf-8")
    assert out_comments.exists()


def test_clone_clusters_include_avg_similarity_in_prompt(tmp_path: Path) -> None:
    """Clone clusters in build_review_prompt must report non-zero avg_similarity."""
    for idx in (1, 2):
        (tmp_path / f"dup_{idx}.py").write_text(
            f"def compute_clone_{idx}(a: int, b: int) -> int:\n"
            "    if a > 0:\n        return a + b\n    return b\n",
            encoding="utf-8",
        )
    report = analyze_repository(tmp_path, clone_threshold=0.80)
    assert len(report.clone_clusters) >= 1
    assert report.clone_clusters[0].get("avg_similarity", 0.0) >= 0.80
    prompt = build_review_prompt(report)
    assert "avg similarity 0.000" not in prompt


def test_cli_outputs_separate_deterministic_and_detailed_gemini_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI must support separate deterministic and standalone detailed Gemini report files."""
    from google import genai

    captured_configs: list[Any] = []

    class _FakeModels:
        def generate_content(self, **kwargs: Any) -> Any:
            captured_configs.append(kwargs.get("config"))
            return type(
                "Resp",
                (),
                {
                    "text": (
                        "### 1. Executive Risk Assessment\n"
                        "Detailed findings for `app.py:1` (`alpha`)."
                    )
                },
            )()

    class _FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            self.models = _FakeModels()

    monkeypatch.setattr(genai, "Client", _FakeClient)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "app.py").write_text("def alpha(x: int) -> int:\n    return x + 1\n", encoding="utf-8")

    out_md = tmp_path / "combined.md"
    out_det = tmp_path / "deterministic.md"
    out_gemini = tmp_path / "gemini_detailed.md"

    rc = cli_main(
        [
            "--mode",
            "repo",
            "--repo",
            str(repo),
            "--out-md",
            str(out_md),
            "--out-deterministic-md",
            str(out_det),
            "--out-gemini-md",
            str(out_gemini),
        ]
    )
    assert rc == 0
    assert out_md.exists()
    assert out_det.exists()
    assert out_gemini.exists()

    combined_text = out_md.read_text(encoding="utf-8")
    det_text = out_det.read_text(encoding="utf-8")
    gemini_text = out_gemini.read_text(encoding="utf-8")

    assert "Gemini Review" in combined_text
    assert "Astraea Periodic Repository Review" in combined_text
    assert "Gemini Review" not in det_text
    assert "Astraea Periodic Repository Review" in det_text
    assert "Detailed findings for `app.py:1`" in gemini_text
    assert "Astraea Periodic Repository Review" not in gemini_text
    assert captured_configs and getattr(captured_configs[-1], "max_output_tokens", None) is None

    # When no Gemini credentials are set, --out-gemini-md still emits an informative status note.
    monkeypatch.delenv("GEMINI_API_KEY")
    out_gemini_no_key = tmp_path / "gemini_no_key.md"
    rc_no_key = cli_main(
        [
            "--mode",
            "repo",
            "--repo",
            str(repo),
            "--out-md",
            str(out_md),
            "--out-gemini-md",
            str(out_gemini_no_key),
        ]
    )
    assert rc_no_key == 0
    assert "unavailable" in out_gemini_no_key.read_text(encoding="utf-8").lower()
