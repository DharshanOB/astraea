"""CLI entrypoint for Astraea, the 4-Layer PR & Periodic Repository Review Bot."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from pr_bot.analyzer import (
    PRReviewReport,
    RepoReviewReport,
    analyze_git_pr,
    analyze_repository,
    resolve_merge_base,
    validate_git_ref,
)
from pr_bot.gemini_reviewer import (
    generate_vertex_gemini_review,
    has_ai_credentials,
    verify_review_claims,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run PR or full-repo 4-layer analysis and write Markdown + optional inline JSON."""
    parser = argparse.ArgumentParser(
        prog="astraea-review",
        description=(
            "Compiler-aware 4-layer GitHub PR & periodic repo reviewer "
            "(AST + CPG + LLVM IR + Machine Code)."
        ),
    )
    parser.add_argument("--mode", choices=("pr", "repo"), default="pr", help="Review mode.")
    parser.add_argument("--repo", default=".", help="Path to git repository root.")
    parser.add_argument("--base", default="HEAD~1", help="Base git ref/SHA (PR mode).")
    parser.add_argument("--head", default="HEAD", help="Head git ref/SHA (PR mode).")
    parser.add_argument("--compile-commands", default=None, help="Optional compile_commands.json.")
    parser.add_argument("--out-md", default="pr_review_summary.md", help="Output Markdown path.")
    parser.add_argument(
        "--out-deterministic-md",
        default=None,
        help="Optional output path for the deterministic 4-layer Markdown report only.",
    )
    parser.add_argument(
        "--out-gemini-md",
        default=None,
        help="Optional output path for the standalone detailed Gemini Markdown report.",
    )
    parser.add_argument(
        "--out-comments",
        default=None,
        help="Output inline comments JSON (defaults to pr_review_comments.json in pr mode).",
    )
    parser.add_argument(
        "--vertex-project",
        default=None,
        help="Optional Google Cloud project ID for Vertex AI IAM/WIF auth.",
    )
    parser.add_argument("--vertex-location", default="global", help="Vertex AI region/endpoint.")
    parser.add_argument(
        "--gemini-model", default="gemini-4-hydrogen", help="Gemini model name."
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional prior baseline snapshot JSON for run-to-run diffing (--mode repo).",
    )
    parser.add_argument(
        "--out-baseline",
        default=None,
        help="Optional output path to save current baseline snapshot JSON (--mode repo).",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=100_000,
        help="Approximate token budget for Gemini review context (default: 100000).",
    )
    args = parser.parse_args(argv)

    report: PRReviewReport | RepoReviewReport
    diff_text = ""
    if args.mode == "repo":
        report = analyze_repository(
            repo_path=args.repo,
            compile_commands_path=args.compile_commands,
            baseline_path=args.baseline,
        )
        if args.out_baseline:
            Path(args.out_baseline).write_text(
                json.dumps(report.snapshot(), indent=2), encoding="utf-8"
            )
        inline_comments = []
    else:
        base_ref = validate_git_ref(args.base)
        head_ref = validate_git_ref(args.head)
        report = analyze_git_pr(
            repo_path=args.repo,
            base_ref=base_ref,
            head_ref=head_ref,
            compile_commands_path=args.compile_commands,
        )
        inline_comments = report.to_github_review_comments()
        repo_root = Path(args.repo).resolve()
        effective_base = resolve_merge_base(repo_root, base_ref, head_ref)
        diff_proc = subprocess.run(
            ["git", "-C", str(repo_root), "diff", effective_base, head_ref],
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_proc.returncode != 0:
            logger.warning(
                "git diff failed (rc=%d): %s",
                diff_proc.returncode,
                diff_proc.stderr.strip(),
            )
            diff_text = ""
        else:
            diff_text = diff_proc.stdout

    deterministic_md = report.to_markdown()
    markdown = deterministic_md
    gemini_md = (
        f"# Gemini Detailed Code Review ({args.gemini_model})\n\n"
        "> **Note:** AI review unavailable "
        "(no Antigravity/ADC token, Vertex project, or Gemini API key configured).\n"
    )
    if has_ai_credentials(vertex_project=args.vertex_project):
        try:
            ai_review = generate_vertex_gemini_review(
                report=report,
                diff_text=diff_text,
                project=args.vertex_project,
                location=args.vertex_location,
                model=args.gemini_model,
                max_prompt_tokens=args.max_prompt_tokens,
            )
        except Exception as exc:
            logger.warning(
                "Gemini review generation failed (%s); writing deterministic report", exc
            )
            if os.environ.get("GITHUB_ACTIONS") == "true":
                print(
                    f"::warning::Gemini review generation failed ({type(exc).__name__}): {exc}",
                    file=sys.stderr,
                )
            note = (
                f"> **Note:** AI review unavailable "
                f"(`{type(exc).__name__}`); deterministic 4-layer report emitted.\n"
            )
            markdown = f"{markdown}\n{note}"
            gemini_md = f"# Gemini Detailed Code Review ({args.gemini_model})\n\n{note}"
        else:
            if ai_review:
                verified_review, _ = verify_review_claims(ai_review, report)
                deterministic_md = report.to_markdown()
                gemini_md = (
                    f"# Gemini Detailed Code Review ({args.gemini_model})\n\n"
                    f"{verified_review}\n"
                )
                markdown = (
                    f"### Gemini Review ({args.gemini_model})\n\n"
                    f"{verified_review}\n\n---\n\n{deterministic_md}"
                )

    Path(args.out_md).write_text(markdown, encoding="utf-8")
    if args.out_deterministic_md:
        Path(args.out_deterministic_md).write_text(deterministic_md, encoding="utf-8")
    if args.out_gemini_md:
        Path(args.out_gemini_md).write_text(gemini_md, encoding="utf-8")
    comments_out: str | None = (
        args.out_comments
        if args.mode == "repo"
        else (args.out_comments or "pr_review_comments.json")
    )
    if comments_out is not None:
        Path(comments_out).write_text(json.dumps(inline_comments, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
