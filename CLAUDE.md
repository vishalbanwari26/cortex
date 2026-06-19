# CLAUDE.md

Project context for Claude Code. Read this before making changes.

## What this is

Cortex is a cognitive-loop orchestrator for embodied agents, built as a
portfolio piece for an Agentic AI Engineer application. It takes a goal and a
scene, perceives the scene (with a VLM when available), plans a multi-step task,
dispatches steps to modular sub-agents and skills, and adapts in real time when
a step fails. Every stage streams as an event.

It is a **sandbox**: the skills are simulated, not wired to a robot. The value
on display is the orchestration architecture, not robotics hardware. Keep this
framing honest in any docs or commit messages. Do not describe it as controlling
a real robot.

## Architecture (where things live)

- `cortex/orchestrator.py` — the main agent. Owns the loop: perceive -> plan ->
  execute step-by-step -> critique on failure -> replan (bounded by
  `max_replans`). Read this first; it is short on purpose.
- `cortex/agents/` — single-responsibility sub-agents, all behind `Agent`
  (base.py): `perception.py` (VLM -> Scene), `planner.py` (goal+scene -> steps),
  `actor.py` (dispatch a step to a skill; not an LLM call), `critic.py` (assess
  failure -> replan/abort).
- `cortex/skills/` — `registry.py` (Skill + SkillRegistry) and `simulated.py`
  (navigate, grasp, place, scan), which read/mutate a `WorldState`.
- `cortex/llm/` — the provider-agnostic boundary. `base.py` defines `LLMClient`;
  concrete clients: `mock_client.py` (offline, deterministic), `anthropic_client.py`
  (vision/VLM), `groq_client.py` (GPT-OSS 120B, text-only). Agents depend ONLY on
  the `LLMClient` interface, never on a vendor SDK directly.
- `cortex/events.py` — typed events + `EventBus` for real-time feedback.
- `cortex/cli.py` — `python -m cortex.cli`, `--provider {mock,anthropic,groq}`.
- `cortex/capture.py` — renders a styled terminal still/GIF/transcript from a
  run, labeled with the provider that produced it.
- `tests/` — all run offline on the mock client (no key, no network).
- `docs/` — architecture.png/svg and demo.gif used by the README.

## Conventions (follow these)

- Python 3.10+, fully type-hinted, dataclasses for data, ABCs for contracts.
- Never import a provider SDK in the agents or orchestrator. New models are new
  `LLMClient` implementations, lazily importing their SDK inside `__init__`.
- Agents prompt for JSON and parse via `Agent._parse_json` (tolerates code
  fences and surrounding prose). Keep new agents on that pattern.
- Skills register in `default_registry()` and operate on `WorldState`. Adding a
  capability should not require touching the orchestrator or planner.
- Keep the orchestrator thin; intelligence lives in the agents.

## Commands

- Install (dev): `pip install -e ".[dev]"`
- Test: `pytest -q`  (expect 11 passing, all offline)
- Run offline: `python -m cortex.cli "Put the red mug in the cupboard."`
- Run on Groq (text): `pip install -e ".[groq]"` then
  `python -m cortex.cli "<goal>" --provider groq --note "<scene description>"`
- Run on Anthropic (VLM): `pip install -e ".[live]"` then
  `python -m cortex.cli "<goal>" --provider anthropic --image examples/scene.jpg`
- Capture assets: `pip install -e ".[assets]"` then `python -m cortex.capture ...`

## Secrets

Keys come from `ANTHROPIC_API_KEY` and `GROQ_API_KEY` in the environment only.
Never hard-code a key, never commit one, never print one. `.env` is gitignored.

## Honesty rules (important)

- Do not invent capabilities, benchmarks, or metrics anywhere (code, README,
  commits). State only what the code actually does.
- gpt-oss-120b is text-only; it cannot do image perception. Do not let docs imply
  otherwise.
- Captured artifacts must stay labeled by their real provider (mock = "offline
  deterministic demo"; live = "live run | provider | model").

## Style for generated prose (README, comments, commits)

Plain, direct language. No em-dashes. Short sentences over clever ones.

## Likely next tasks

- Run the live providers and capture real artifacts.
- If gpt-oss wraps JSON in reasoning text, add Groq structured-output mode
  (response_format / JSON schema) in `groq_client.py`.
- Optional: a Groq vision client (e.g. Llama 4 Scout) for a fully-Groq vision
  path, as a separate `LLMClient`.
- Optional: replace simulated skills with a real robot/ROS bridge behind the
  same registry interface.
