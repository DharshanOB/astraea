# Astraea Review Bot (`pr_bot`)

`pr_bot` powers Astraea's automated pull request reviews and scheduled full-repository health audits.

## Components

- `analyzer.py`: Implements `analyze_pr_directories`, `analyze_git_pr`, and `analyze_repository`. Collects AST complexity, CPG blast radius, clustered clones, LLVM IR / Python bytecode lowering, top hotspot sources, and bounded 1-hop CPG caller/callee context.
- `gemini_reviewer.py`: Builds grounded prompts (`build_review_prompt`, `build_pr_reply_prompt`), runs single-call batched Stage-4 helper classification (`classify_custom_guard_helpers`), and calls Gemini Developer API / OpenCode Console / Vertex AI / `agy` CLI with `retryDelay`-aware `429` backoff (`generate_vertex_gemini_review`, `generate_pr_reply`).
- `cli.py`: CLI entrypoint (`astraea-review`).

## GitHub Actions Setup

Four AI authentication modes are supported in `.github/workflows/pr-review.yml`, `.github/workflows/pr-chat.yml`, and `.github/workflows/periodic-review.yml` (default model: `gemini-3.5-flash-lite`):

1. **OpenCode Console / Go / Free Models (`OPENCODE_API_KEY`, keyless OIDC or API key, preferred over env `GEMINI_API_KEY`):**
   - Set `OPENCODE_API_KEY` (for example via GitHub Actions OIDC token exchange against `https://opencode.ai/console`) to route Gemini/Go inference through `https://opencode.ai/inference/google/v1beta` (`opencode-go/*` models automatically route to `https://opencode.ai/inference/go/google/v1beta`; override with `OPENCODE_BASE_URL`).
   - Free-tier OpenCode models (such as `opencode/big-pickle` or `opencode/deepseek-v4.1-flash-free`) automatically run inside an isolated `opencode run --standalone --format json` sandbox when the `opencode` CLI is installed.
2. **Gemini API Key (`GEMINI_API_KEY`, fallback or direct GitHub-hosted CI):**
   - Set repository secret `GEMINI_API_KEY` to call the Gemini Developer API (`v1beta`) directly via `google-genai`.
3. **Google Cloud Vertex AI (`ADC` / `WIF`):**
   - Set `AGY_TOKEN_JSON` to a GCP authorized-user/service-account JSON with a billable `project_id` (restored to `~/.config/gcloud/application_default_credentials.json`) or configure `GCP_PROJECT_ID`, `GCP_LOCATION`, and `GOOGLE_GENAI_USE_VERTEXAI` with Workload Identity Federation.
4. **Antigravity Session (`AGY_TOKEN_JSON` / `agy` CLI on interactive or self-hosted runners):**
   - `.github/scripts/restore_token.py` writes consumer OAuth tokens (`aicode-consumers`) to `~/.gemini/antigravity-acp/acp_token.json` so local/self-hosted `agy` sessions can run headless `stream-json`.
5. **Interactive PR Replies (`--mode reply` & `.github/workflows/pr-chat.yml`):**
   - Developers with `OWNER`, `MEMBER`, or `COLLABORATOR` association can mention `@astraea` or `/astraea` in a PR conversation comment or inline review thread to get a grounded root-cause explanation and fix/patch suggestion.
6. Optional repository variable:
   - `GEMINI_MODEL`: Override the default Gemini model name (default `gemini-3.5-flash-lite`).
7. Add `.github/workflows/pr-review.yml`, `.github/workflows/pr-chat.yml`, and `.github/workflows/periodic-review.yml` to your target repository.
