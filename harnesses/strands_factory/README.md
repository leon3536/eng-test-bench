# Strands harness factory

The bench's strands arms run Harbor's built-in `strands` agent with this
factory. The factory builds the agent with
[`strands-harness`](https://pypi.org/project/strands-harness/), the Strands
team's published harness.

**Why this route.** Harbor's strands adapter installs strands-agents in the
task container and runs a factory you supply; it has no agent of its own. The
Strands team's figures (S8, "We ran 94,000 trials to measure the cost of
harness choice", and the Terminal-Bench numbers from the "Introducing Strands
harness" launch) were measured with `create_harness()` running through this
same adapter. SSA (simple-strands-agent) is a different, earlier research
harness. It crashes on Harbor 0.25, cannot run its commands on Windows, and is
not what S8 measured, so the bench does not use it.

## Pinned versions

| Package | Version | Where it is set |
|---|---|---|
| strands-harness | 0.3.2 | `pyproject.toml` in this folder |
| strands-agents | 1.57.2 | the arm's Harbor option `version` |
| Harbor | 0.25.0 | the bench's `requirements.txt` |

strands-harness 0.3.2 requires `strands-agents>=1.57.2`. Harbor installs
`strands-agents[all]==<version>` and defaults to 1.54.0, which cannot be
resolved together with it. Every arm that uses this factory must therefore
set `version: "1.57.2"`.

## How an arm uses it

```yaml
- name: strands
  agent: strands
  provider: anthropic
  model: claude-opus-5-5
  effort: omit                 # Harbor's adapter has no effort option; effort goes in factory_kwargs
  options:
    factory: bench_strands_factory:create_agent
    factory_path: harnesses/strands_factory
    version: "1.57.2"
    factory_kwargs:
      model_spec: anthropic/claude-opus-5-5
      effort: medium
```

`factory_path` is relative to the project root, which is Harbor's working
directory. For each trial Harbor does the following:
1. It copies this folder into the container at
   `/installed-agent/strands/factory`. It skips `.env*`, `.git`, `.venv` and
   `__pycache__`, and it refuses symbolic links.
2. It creates a Python 3.12 venv with uv and installs
   `strands-agents[all]==1.57.2` and `-e` this folder. The `-e` install is
   what pulls in strands-harness.
3. It runs its runner with `PYTHONPATH` set to the folder. The runner calls
   `create_agent(sandbox, **factory_kwargs)` and adds `model=` (a Model that
   Harbor built from `-m`) unless `factory_kwargs` already has `model`.

## What `create_agent` does

`create_agent(sandbox, model=None, *, model_spec=None, effort="auto", **harness_kwargs)`
returns the `strands.Agent` from `create_harness()`, with `sandbox` passed
through so every tool runs in the task container.

- **With `model_spec`** (`"provider/name"`): strands-harness builds the model
  itself. It applies `effort` and sets up prompt caching (cache points and
  cached tool definitions on Anthropic). Harbor's prebuilt `model` is ignored.
  The default profile uses this. `ANTHROPIC_API_KEY` reaches the runner for
  anthropic arms, and the Anthropic client reads it from the environment.
- **`model_spec` must be the arm's own model.** Harbor's runner sets
  `HARBOR_MODEL` to the arm's `-m` value (for example
  `anthropic/claude-opus-5-5`). If `model_spec` differs from it (ignoring
  case), `create_agent` raises `ValueError` at construction, before any model
  call, and the trial fails with that message. Harbor, the ledger and the run
  budget price a trial at the arm's model, so a `model_spec` left over from a
  copied arm or design would otherwise run one model and be billed as
  another. A string `model` in `factory_kwargs` is checked the same way.
- **Without it:** Harbor's prebuilt `model` is used, with effort `auto`. If
  `effort` is set to anything else, the factory logs one warning line saying
  it cannot be applied. The dev profile works this way.
- **`effort`:** a strands-harness level: `auto`, `off`, `minimal`, `low`,
  `medium`, `high`, `xhigh` or `max`. `auto` is the provider's recommended
  level, which is `high` on Anthropic. A level the provider does not offer
  fails the trial at construction. A YAML `off` (read as false) is treated as
  `off`, and null as `auto`.
- **Anything else** in `factory_kwargs` goes to `create_harness()` unchanged,
  for example `builtin_tools: {subagent: false}`.

### Defaults kept and changed

Every strands-harness default is kept, because those defaults are what S8
measured:
- Tools: shell, read, write, edit, web_fetch, programmatic_tool_caller and
  subagent. web_search is added too, but only where the provider has native
  search (Anthropic, given a `model_spec`).
- The `todos` and `environment` plugins, the `auto` context manager, prompt
  caching, skills from `./.agent/skills` if that folder exists, and no
  interventions.

One default changed: where session and memory files go. The factory passes
`session` and `memory` folders under `/tmp/bench-strands-harness`.
strands-harness would otherwise write `./.agent/sessions` and
`./.agent/memory` into the runner's working directory. That directory is the
task's own (for example `/app`), and the task's tests inspect it. The features
and their model calls are the same; only the location moves. An arm can still
set `session` or `memory` in `factory_kwargs`, and its value wins.

## Verification done (2026-10-29)

Nothing below installed or ran the downloaded packages.

- **Download and read.** `pip download --no-deps` of `strands-harness==0.3.2`
  and `strands-agents==1.57.2` into a new folder under `%TEMP%`. Both wheels
  were unzipped and read.
- **`create_harness()`.** It is keyword-only: `model`, `effort`,
  `instructions`, `tools`, `plugins`, `mcp_servers`, `builtin_tools`,
  `background_tasks`, `caching`, `context_manager`, `session`, `skills`,
  `memory`, `builtin_plugins`, `interventions`, plus `**agent_kwargs`.
  - `sandbox` travels in `agent_kwargs` to `strands.Agent(sandbox=...)`, which
    accepts any `strands.sandbox.Sandbox`. Harbor's `TaskLocalSandbox` is a
    `PosixShellSandbox`, so it qualifies.
  - The harness then drops the tools the sandbox vends (`sandbox_shell`,
    `sandbox_file_editor`) and routes its own tools through `agent.sandbox`.
  - Subagent children get the same sandbox.
  - So the runner contract works as is, and no fallback factory was needed.
- **`models.resolve_model`.** A `"provider/name"` string gets effort mapped
  and caching configured. The providers are bedrock, bedrock-mantle,
  anthropic, openai, google, ollama and litellm.
  - On `anthropic/claude-opus-5-5` this means adaptive thinking with
    summarized display, `output_config.effort`, max_tokens 128000,
    `CacheConfig(strategy="auto")` and native web search.
  - A prebuilt Model instance is used verbatim: a non-`auto` effort and
    caching are ignored with a logged warning.
- **Imports.** An AST check against strands-agents 1.57.2 found every import
  that Harbor's runner makes:
  - `strands.Agent`, `SandboxTimeoutError`;
  - `strands.hooks.MessageAddedEvent`;
  - `strands.sandbox` `ExecutionResult`, `PosixShellSandbox`, `StreamChunk`,
    `StreamType`;
  - `strands.tools.mcp.MCPClient`;
  - `strands.vended_tools` `make_file_editor` and `make_shell`, with
    `DEFAULT_FILE_EDITOR_DESCRIPTION` and `SANDBOX_SHELL_DESCRIPTION`.

  The same check found all 52 strands imports that strands-harness 0.3.2
  makes.
- **Harbor 0.25 adapter probe.** The adapter was constructed in process,
  with a fake key and python-dotenv disabled, and no trial was run.
  - It stages exactly `pyproject.toml` and `bench_strands_factory/__init__.py`,
    plus this README.
  - It adds `-e /installed-agent/strands/factory` to the install and asks for
    `strands-agents[all]==1.57.2`.
  - The runner env has `ANTHROPIC_API_KEY` for an anthropic arm and
    `OPENROUTER_API_KEY` for an OpenRouter arm. For OpenRouter, Harbor's
    runner builds a LiteLLM model.

**Not verified yet:** no trial has run with this factory. The container
install (strands-agents[all], strands-harness and its `pydantic-monty`
dependency, all from PyPI) and the first model call have not been exercised.
Run a 1-task pilot before a sample.

## Known gaps

- **Dev arms (OpenRouter).** These use the model Harbor builds, so
  strands-harness applies neither effort nor caching. It logs a warning for
  caching. Native web search is off, and web_fetch summaries and memory
  extraction run on the main model.
  - A `model_spec` would not help here. strands-harness's `litellm/...`
    provider has no effort levels, and caching on it is server-side anyway.
    A `litellm/...` spec also differs from the arm's `openrouter/...` model,
    so `create_agent` refuses it.
- **Token accounting.** Harbor counts the main agent loop only (the runner's
  accumulated usage). These calls are not in a trial's tokens or computed
  cost:
  - subagent children, which run on the same model;
  - the web_fetch summarizer and memory extraction, which run on Claude Haiku
    4.5 (`claude-haiku-4-5-20251001`) when the main model is Anthropic;
  - Anthropic's per-search fee for native web search.

  When the agent uses these, the strands arm's tokens and cost are a lower
  bound.
- **Benchmark.** S8 ran Terminal-Bench 2.1. The bench runs terminal-bench@2.0.
- **Effort.** S8 ran at high effort. The default profile passes `medium` to
  match `run.effort`. If you change `run.effort`, change the strands arm's
  `factory_kwargs.effort` too; nothing keeps them in step automatically. To
  match S8 instead, set `effort: auto` (or `high`) in `factory_kwargs`.
- **Limits per trial.** The adapter takes no dollar cap and no turn cap. A
  strands trial is bounded only by its task timeout. The run budget sees its
  cost when it finishes, so trials in flight can take a run past its budget
  (see How a run works in the bench's README). Its subagent and web_fetch
  calls are not in Harbor's token counts, so its recorded spend is a lower
  bound when it uses them.
