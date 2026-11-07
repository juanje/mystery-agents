# Handoff: OpenAI Decisions API for game evaluations

Audience: the agent adding OpenAI support for text generation. The goal is to judge whether the
Decisions API fits the quality problems you are seeing in your OpenAI runs, and where. Nothing here has
been implemented or tried against the real API yet. Treat the integration ideas as hypotheses.

Source: https://developers.openai.com/api/docs/guides/decisions (read 2026-11-07).

## 1. What the API is

- `POST /v1/decisions`. Only model: `gpt-6-luna`. The docs say it is about 10x faster than the
  Responses API.
- Pricing: $0.10 per 1M uncached input tokens. Output tokens are free. Cache hits are discounted and
  cache writes are charged.
- Prompt caching is on by default, with an implicit breakpoint at the end of the shared `input`. The
  reusable prefix must be at least 1,024 tokens. `prompt_cache_key` and `prompt_cache_options` are also
  supported.
- **Request**: one shared `input` (a string or a list of messages) plus 1–400 `questions`. Each question
  has a `type`, a unique `name` and `instructions`. User messages accept text, images and `input_json`.
  Developer and system messages accept text only. A question's instructions can carry their own image.
- **Question types**:
  - `predicate`: `probability` (0–1) that a condition holds.
  - `choice`: picks one of the supplied `choices` (`value` + `description`). It returns `choice`, per-choice
    `probabilities` and `confidence`.
  - `score`: rates against 2–10 ordered `levels` (`label` + `description`). It returns a probability-weighted
    `score` (it can fall between levels), per-level `probabilities` and `confidence`.
- **Response**: an `answers` list, one per question, matched by `name`. An answer can be a `refusal`.
- **Not supported**: free-form text or explanations, extraction into your own schema, tool calls,
  `file_id`/audio input, and questions that depend on each other within one request (use separate
  requests).
- **Not documented**: maximum input size, maximum number of choices, language support, determinism or
  seeds, and batch mode.

SDK: the installed `openai` 3.46.0 (pulled in by `langchain-openai`) already exposes it. The signature was
checked locally:

```python
client.decisions.create(*, input, model, questions, prompt_cache_key=..., prompt_cache_options=...,
                        safety_identifier=..., timeout=...) -> Decision
```

LangChain does not wrap this endpoint, so call the `openai` client directly. The fields of the response
object come from the docs and have not been checked against a real response.

## 2. Key constraint: no rationale

V1, V2 and V3 return `issues` and `suggested_fixes` as free text, and the retry loops feed that text back
to the generator. Decisions cannot produce it, so it **cannot replace** those validators. It works when
each question targets one concrete item (a clue, a guiding question, a character's timeline). Code can
then write the `ValidationIssue.description` from a template, the same way the rule checks in
`find_rule_violations` (`src/mystery_agents/agents/v3_content_validator.py`) already do.

## 3. Where it could help, ranked by expected value

### 3.1 Simulated detective: a difficulty and solvability metric (highest value)

Problem: reviewers find games too easy or unfair, and nothing in the pipeline measures that.

How: build the input from **only what players see**: clue texts, suspects' public sheets and
observations, and the floor plan. Leave out the solution and the `incriminates`/`exonerates` tags. Then
ask one `choice` question, "Based only on this evidence, who is the killer?", with one choice per
suspect.

- P(killer) close to 1: too easy. Very low, or spread evenly across suspects: unsolvable or too
  ambiguous. The target band depends on `config.difficulty`.
- **Leave-one-out**: one request per clue with that clue removed. The drop in P(killer) shows how
  decisive each clue is. A clue whose removal changes nothing is filler. A clue that takes P(killer) from
  0.2 to 0.95 on its own is the "solves the case alone" problem.
- It is cheap enough to run on every attempt and to compare across runs and providers.

Start it as a logged metric in V3 (`v3_content_validator.py`, `ContentValidatorAgent.run`). Turn it
into a gate only after calibrating thresholds on several games.

### 3.2 Over-obvious clues

Today: `find_over_obvious_clues` relies on the `incriminates`/`exonerates` lists that the generating LLM
tags itself, and those tags can be wrong.

How: one `predicate` per clue, all in one request: "Given the suspects and the crime, does this clue
alone identify a single suspect as the killer?" The shared input is the case summary and the suspects,
and it is cached. A probability above a threshold creates an `over_obvious` `ValidationIssue` with the
clue id in `related_ids`.

### 3.3 Spoilers in player-facing text

Today: `drop_revealing_questions` and `find_killer_in_guiding_questions` match names and minutes, so
they miss questions that point at the killer without naming them (role, initials, an object only they
carry). Clue titles and the Act 2 intro have the same weakness (`find_suspect_names_in_titles`).

How: one `predicate` per guiding question or title, with the solution in the input: "Would a player
reading only this text be steered towards <killer>?" Keep the deterministic checks and add this one on
top.

### 3.4 Targeted retries (open thread: A8 rewrites everything on retry)

Because each answer is tied to one clue id, question or character, the failing items are known exactly.
A8 can then be asked to edit only those items and keep the rest verbatim. Same idea for A7.5: one
`predicate` per character timeline ("Is this timeline internally consistent with the master timeline
and the floor plan?"). This targets the contradictions that V3 finds and V2 misses.

### 3.5 Pre-filter before the full V3 call

If every per-item predicate is clean, the expensive V3 LLM review could be skipped or shortened. This is
only worth it after 3.1–3.3 prove reliable.

### 3.6 Portrait checks (images stay on Gemini)

Images are allowed in the input and inside a question's instructions. Possible checks: the portrait
matches the character description (gender, age, era costume); no visible defects; not too similar to
another portrait (put the other image in the question). Relevant to `a3_5_character_images.py`.

### 3.7 Cheap version of the eval rubric

`docs/EVAL_RUBRIC.md` has six 1–5 dimensions. Each could become a `score` question with the rubric text
as level descriptions. That gives a fast numeric signal between full reviewer-agent batches. It would
not replace them, because the rubric also asks for quoted problems.

## 4. Usage sketch (untested)

```python
from openai import OpenAI

client = OpenAI()

decision = client.decisions.create(
    model="gpt-6-luna",
    input=[
        {"role": "developer", "content": "You are a guest at a murder mystery party. Judge only from the evidence given."},
        {"role": "user", "content": [{"type": "input_text", "text": player_visible_evidence}]},
    ],
    questions=[
        {
            "type": "choice",
            "name": "accused",
            "instructions": "Based only on this evidence, who is the killer?",
            "choices": [
                {"value": char.id, "description": f"{char.name}, {char.role}"} for char in state.characters
            ],
        },
        *(
            {
                "type": "predicate",
                "name": f"solves_alone_{clue.id}",
                "instructions": f"Does clue '{clue.title}' on its own point to a single suspect as the killer?",
            }
            for clue in state.clues
        ),
    ],
    prompt_cache_key=f"mystery-{state.meta.id}",
)

for answer in decision.answers:
    ...  # match by answer.name; handle answer.type == "refusal"
```

Check the exact content-part type names (`input_text`, `input_json`) against the API reference before
using this. `state.meta.id` is illustrative.

## 5. Risks and open questions to verify

1. **Refusals**: the content is about murders and poisonings. Measure the refusal rate on real game
   states first. Gemini already refused some prompts in this project.
2. **Spanish**: the docs say nothing about language. Compare results on Spanish and English versions of
   the same game.
3. **Input size**: a full game state is large and no limit is documented. Measure tokens on a real
   state.
4. **Calibration**: probabilities need thresholds per difficulty level. Run several saved games (`save/`)
   where reviewer verdicts are known, and check whether the numbers separate good games from bad ones.
5. **Stability**: run the same request several times and check the variance. Determinism is not
   documented.
6. **Optional dependency**: it needs an OpenAI key. The checks must be skippable so that a Gemini-only
   setup still works, and they belong behind config like the provider selection you are adding.

## 6. Suggested first experiment

Take 3–5 existing generated games whose reviewer scores are known. Run 3.1 (simulated detective plus
leave-one-out) and 3.2 offline from a small script. Then check two things: does P(killer) track the
reviewer's difficulty and solvability scores, and do the flagged clues match the ones reviewers called
too obvious? If they do, wire 3.1 into V3 as a logged metric first.
