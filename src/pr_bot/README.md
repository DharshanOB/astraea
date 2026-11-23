# Astraea Review Bot (`pr_bot`)

`pr_bot` powers Astraea's automated pull request reviews and scheduled full-repository health audits.

## Components

- `analyzer.py`: Implements `analyze_pr_directories`, `analyze_git_pr`, and `analyze_repository`. Collects AST complexity, CPG blast radius, clustered clones, LLVM IR / Python bytecode lowering, top hotspot sources, and bounded 1-hop CPG caller/callee context.
- `gemini_reviewer.py`: Builds grounded prompts (`build_review_prompt`) and calls Vertex AI / Gemini (`generate_vertex_gemini_review`).
- `cli.py`: CLI entrypoint (`astraea-review`).

## GitHub Actions Setup

Three AI authentication modes are supported in `.github/workflows/pr-review.yml` and `.github/workflows/periodic-review.yml` (default model: `gemini-3.8-flash`):

1. **Antigravity Session Token (`AGY_TOKEN_JSON`, recommended for personal/Pro/Ultra accounts):**
   - Set repository secret `AGY_TOKEN_JSON` to the Antigravity ACP/OAuth JSON payload.
   - `.github/scripts/restore_token.py` writes consumer OAuth tokens (`aicode-consumers`) to `~/.gemini/antigravity-acp/acp_token.json`, and the workflows install the Antigravity CLI (`agy`) so `astraea-review` invokes `agy --input-format stream-json --output-format stream-json`.
2. **Gemini API Key (`GEMINI_API_KEY`):**
   - Set repository secret `GEMINI_API_KEY` to call the Gemini API directly via `google-genai`.
3. **Google Cloud Vertex AI (`ADC` / `WIF`):**
   - Set `AGY_TOKEN_JSON` to a GCP authorized-user/service-account JSON with a billable `project_id` (restored to `~/.config/gcloud/application_default_credentials.json`) or configure `GCP_PROJECT_ID`, `GCP_LOCATION`, and `GOOGLE_GENAI_USE_VERTEXAI` with Workload Identity Federation.
4. Optional repository variable:
   - `GEMINI_MODEL`: Override the default Gemini model name (default `gemini-3.8-flash`).
5. Add `.github/workflows/pr-review.yml` and `.github/workflows/periodic-review.yml` to your target repository.
