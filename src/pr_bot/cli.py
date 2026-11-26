"""CLI entrypoint for Astraea, the 4-Layer PR & Periodic Repository Review Bot."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

from pr_bot.analyzer import (
    PRReviewReport,
    RepoReviewReport,
    analyze_git_pr,
    analyze_repository,
    resolve_merge_base,
    validate_git_ref,
)
from pr_bot.gemini_reviewer import (
    generate_pr_reply,
    generate_vertex_gemini_review,
    has_ai_credentials,
    verify_review_claims,
)

logger = logging.getLogger(__name__)


def _read_file_snippet_from_git(
    repo_root: Path,
    git_ref: str,
    file_path: str,
    line: int | None,
    context_lines: int = 40,
) -> str:
    """Safely read numbered source lines around `line` from `<git_ref>:<file_path>`."""
    raw_path = file_path.strip()
    if not raw_path or raw_path.startswith(("/", "\\")):
        return ""
    posix_path = PurePosixPath(raw_path)
    norm_path = posix_path.as_posix()
    if norm_path in {"", "."} or ".." in posix_path.parts:
        return ""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "show", f"{git_ref}:{norm_path}"],
        shell=False,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        return ""
    all_lines = proc.stdout.splitlines()
    if not all_lines:
        return ""
    if line is None:
        start = 1
        end = min(len(all_lines), context_lines * 2)
    else:
        start = max(1, line - context_lines)
        end = min(len(all_lines), line + context_lines)
    return "\n".join(f"{idx}: {all_lines[idx - 1]}" for idx in range(start, end + 1))


def _run_reply_mode(args: argparse.Namespace) -> int:
    """Answer a developer's PR comment using diff, inline context, and 4-layer evidence."""
    ctx: dict[str, object] = {}
    if args.reply_context_json:
        ctx_path = Path(args.reply_context_json)
        if ctx_path.is_file():
            loaded = json.loads(ctx_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                ctx = loaded

    question = str(ctx.get("question") or args.question or "").strip()
    if not question:
        Path(args.out_md).write_text(
            "> **Note:** No question provided for Astraea reply.\n",
            encoding="utf-8",
        )
        return 0

    prior_review = str(ctx.get("prior_review") or "")
    raw_history = ctx.get("thread_history")
    thread_history: list[dict[str, str]] = []
    if isinstance(raw_history, list):
        for item in raw_history:
            if isinstance(item, dict):
                thread_history.append(
                    {
                        "author": str(item.get("author", "user")),
                        "body": str(item.get("body", "")),
                    }
                )
    raw_file_path = ctx.get("file_path")
    file_path = str(raw_file_path).strip() if raw_file_path else None
    raw_line = ctx.get("line")
    line: int | None = None
    if isinstance(raw_line, int) and not isinstance(raw_line, bool):
        line = raw_line
    elif isinstance(raw_line, str) and raw_line.strip().isdigit():
        line = int(raw_line.strip())
    diff_hunk = str(ctx.get("diff_hunk") or "")
    file_snippet = str(ctx.get("file_snippet") or "")

    repo_root = Path(args.repo).resolve()
    base_ref: str | None = None
    head_ref: str | None = None
    try:
        base_ref = validate_git_ref(args.base)
        head_ref = validate_git_ref(args.head)
    except ValueError as exc:
        logger.warning("Invalid git ref in reply mode (%s)", exc)

    diff_text = ""
    if base_ref and head_ref:
        effective_base = resolve_merge_base(repo_root, base_ref, head_ref)
        diff_proc = subprocess.run(
            ["git", "-C", str(repo_root), "diff", effective_base, head_ref],
            shell=False,
            capture_output=True,
            text=True,
            check=False,
        )
        if diff_proc.returncode == 0:
            diff_text = diff_proc.stdout
        else:
            logger.warning(
                "git diff failed in reply mode (rc=%d): %s",
                diff_proc.returncode,
                diff_proc.stderr.strip(),
            )

    if file_path and not file_snippet:
        file_snippet = _read_file_snippet_from_git(
            repo_root, head_ref or "HEAD", file_path, line
        )

    pr_report: PRReviewReport | None = None
    if not prior_review.strip() and base_ref and head_ref:
        try:
            pr_report = analyze_git_pr(
                repo_path=args.repo,
                base_ref=base_ref,
                head_ref=head_ref,
                compile_commands_path=args.compile_commands,
            )
        except Exception as exc:
            logger.warning("PR analysis skipped in reply mode (%s)", exc)

    if has_ai_credentials(vertex_project=args.vertex_project):
        try:
            reply_md = generate_pr_reply(
                question=question,
                diff_text=diff_text,
                prior_review=prior_review,
                thread_history=thread_history,
                file_path=file_path,
                line=line,
                diff_hunk=diff_hunk,
                file_snippet=file_snippet,
                report=pr_report,
                project=args.vertex_project,
                location=args.vertex_location,
                model=args.gemini_model,
                max_prompt_tokens=args.max_prompt_tokens,
            )
        except Exception as exc:
            logger.warning("PR reply generation failed (%s)", exc)
            if os.environ.get("GITHUB_ACTIONS") == "true":
                print(
                    f"::warning::PR reply generation failed ({type(exc).__name__}): {exc}",
                    file=sys.stderr,
                )
            reply_md = f"> **Note:** Astraea reply unavailable (`{type(exc).__name__}`).\n"
        else:
            if reply_md and pr_report is not None:
                reply_md, _ = verify_review_claims(reply_md, pr_report)
            elif not reply_md:
                reply_md = "> **Note:** Astraea returned an empty reply.\n"
    else:
        reply_md = (
            "> **Note:** Astraea reply unavailable "
            "(no Gemini, OpenCode, Antigravity/ADC, or Vertex credentials configured).\n"
        )

    Path(args.out_md).write_text(reply_md, encoding="utf-8")
    if args.out_comments is not None:
        Path(args.out_comments).write_text("[]", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run PR or full-repo 4-layer analysis and write Markdown + optional inline JSON."""
    parser = argparse.ArgumentParser(
        prog="astraea-review",
        description=(
            "Compiler-aware 4-layer GitHub PR & periodic repo reviewer "
            "(AST + CPG + LLVM IR + Machine Code)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("pr", "repo", "reply"),
        default="pr",
        help="Review mode.",
    )
    parser.add_argument("--repo", default=".", help="Path to git repository root.")
    parser.add_argument("--base", default="HEAD~1", help="Base git ref/SHA (PR/reply mode).")
    parser.add_argument("--head", default="HEAD", help="Head git ref/SHA (PR/reply mode).")
    parser.add_argument(
        "--reply-context-json",
        default=None,
        help="Optional JSON context file for --mode reply.",
    )
    parser.add_argument(
        "--question",
        default="",
        help="Developer question for --mode reply when not passed via --reply-context-json.",
    )
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

    if args.mode == "reply":
        return _run_reply_mode(args)

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
        "(no Gemini, OpenCode, Antigravity/ADC, or Vertex credentials configured).\n"
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
