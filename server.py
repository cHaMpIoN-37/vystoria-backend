import os
import json
import re
import time
import random
import traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError(
        "Missing SUPABASE_URL or SUPABASE_SERVICE_KEY. Copy .env.example to "
        ".env and fill in your Supabase project's URL and service_role key "
        "(Project Settings -> API in the Supabase dashboard)."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

app = FastAPI(title="Vystoria Creator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Reference documents can be long; cap what we inline into a prompt so a
# creator pasting a whole novel doesn't blow the context window / cost.
MAX_REFERENCE_CHARS = 12000

# Real-world naming traditions we rotate through to break Gemini/GPT's
# "every fantasy protagonist is Kaelen or Lyra" bias. A fresh random draw
# per generation means back-to-back stories won't share a name pool.
NAMING_TRADITIONS = [
    "Yoruba", "Vietnamese", "Farsi", "Quechua", "Icelandic", "Tamil",
    "Basque", "Amharic", "Māori", "Hungarian", "Georgian", "Uzbek",
    "Ojibwe", "Bengali", "Slovenian", "Malagasy", "Kurdish", "Finnish",
    "Sinhalese", "Zulu", "Mongolian", "Catalan", "Tagalog", "Swahili",
    "Armenian", "Thai", "Croatian", "Punjabi",
]

def pick_naming_pool(k: int = 3) -> str:
    return ", ".join(random.sample(NAMING_TRADITIONS, k))

# Modern, widely-available default model IDs — only used as a fallback when
# the frontend forgets to send `model_name`. These strings drift every few
# months as providers retire models, so keep them fresh. The MENTOR-FACING
# lesson learned: hard-coding a specific model version and shipping it to a
# fresh account will break whenever the vendor deprecates that model for
# new users. The frontend now sends a user-editable model string; these are
# just safety nets.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",     # stable, replaces the 2.5-flash line
    "openai": "gpt-4o",
    "claude": "claude-sonnet-4-6",    # stable Sonnet 4 tier — broadest availability
    "grok":   "grok-2-latest",
}

# Substrings we look for inside an SDK exception message to recognize the
# "the model itself is the problem, not the request" family of failures.
# When we see one of these, we rewrite the error into a message that tells
# the creator to change the Model Name field instead of showing a raw stack.
#
# Add new markers as vendors invent new wordings — one substring hit is enough,
# so being generous here is safe (a false positive just gives the creator a
# clearer error message than a raw stack trace).
MODEL_UNAVAILABLE_MARKERS = (
    "no longer available",
    "not found for api version",
    "was not found",
    "does not exist",
    "invalid model",
    "model not found",
    "unknown model",
    "model_not_found",
    "not supported for this",
    "does not have access to model",
    "the model ",              # narrower than "the model" — avoids matching random prose
    "404 models/",             # google's exact format: "404 models/gemini-x-y is not found..."
    "does not support",        # "model X does not support generateContent"
    "is not available",        # anthropic phrasing
    "you don't have access",   # openai/anthropic phrasing
    "permission denied",       # some vendors use this for access-denied-to-model
)

# Exception CLASS names we treat as "the model is the problem." SDK-agnostic —
# doesn't care whether it's google.api_core.exceptions.NotFound or
# openai.NotFoundError or anthropic.NotFoundError; the shared word is NotFound.
# This is the belt-and-braces layer for when str(e) doesn't include a marker
# above (some SDK versions truncate the message before we see it).
MODEL_UNAVAILABLE_EXCEPTION_TYPES = (
    "NotFound",
    "NotFoundError",
    "BadRequestError",       # openai sometimes uses this for unknown model
    "PermissionDeniedError", # anthropic returns this for model-access-denied
    "InvalidArgument",       # google grpc equivalent
)


def suggested_alternatives(provider: str) -> str:
    """Human-readable list of currently-safe model IDs for a given provider,
    used in error messages so the creator knows exactly what to paste."""
    provider = (provider or "").lower()
    if provider == "gemini":
        return "gemini-3.5-flash, gemini-3.1-flash-lite, gemini-3.7-flash"
    if provider == "openai":
        return "gpt-4o, gpt-4o-mini, gpt-4-turbo"
    if provider == "claude":
        return "claude-sonnet-4-6, claude-opus-4-8, claude-haiku-4-5-20251001"
    if provider == "grok":
        return "grok-2-latest, grok-2-beta"
    return "(unknown provider — check the vendor's docs for current model IDs)"


class ModelUnavailableError(Exception):
    """Raised by call_llm when the vendor rejects the *model itself* (as
    opposed to the request payload). Carries a pre-formatted, creator-facing
    message so the pipeline can log it and give up cleanly instead of
    retrying forever on a name that will never resolve."""


# ==========================================
# 2. DATA MODELS
# ==========================================
class GenerateRequest(BaseModel):
    provider: str      # 'gemini', 'openai', 'grok', 'claude'
    api_key: str       # Custom key provided by the creator
    model_name: str    # Any string the vendor accepts — free-text on the frontend
    title: str
    subtitle: str
    genre: str
    target_length: str # e.g., "8 chapters"
    tone: str
    idea: str | None = None
    reference_text: str | None = None  # optional creator-supplied story doc / outline
    user_id: str

class EvaluateRequest(BaseModel):
    """Lets the creator manually (re-)trigger the AI Judge for an already-generated
    task, e.g. with a different provider/model than was used to write the story."""
    provider: str
    api_key: str
    model_name: str

class TweakSceneRequest(BaseModel):
    """Targeted single-scene rewrite. Only the scene supplied is touched — the
    rest of the story JSON never passes through the model."""
    provider: str
    api_key: str
    model_name: str
    world_bible: str
    scene: dict
    instruction: str

# ==========================================
# 3. PROMPT TEMPLATES
# ==========================================
WORLD_PROMPT = """
You are a master Visual Novel writer. Create a rich, dark, atmospheric Visual Novel world.
**Project Details:**
- Title: {title}: {subtitle}
- Genre: {genre}
- Target length: {target_length} (Full Length)
- Tone: {tone}
{idea_section}
{reference_section}

**Generate the following in structured markdown format:**
1. **Protagonist**
2. **Main Characters** (8-12 total)
3. **Key Locations** (15-20)
4. **Core Rules & Systems**
5. **Major Themes** and **Emotional Arc**
6. **Writing Style Guidelines**

**NAMING RULES (critical — read carefully):**
- Do NOT default to generic fantasy/AI-slop names. AVOID entirely: Kaelen, Kael, Kaelin, Lyra,
  Lyria, Elara, Aria, Aeris, Seraphina, Serana, Thorne, Vance, Ashe, Ash, Rowan, Kieran, Sylas,
  Silas, Nyx, Zephyr, Zephyros, Cassian, Cassia, Alaric, Xander, Draven, Ryker, Elowen. These
  are overused and will make the story feel derivative.
- For this specific story, draw ALL character names from these real-world naming traditions:
  **{naming_pool}**. Match the setting's implied culture where possible; when in doubt, mix
  the three traditions above so the cast feels varied rather than mono-cultural.
- Every character has ONE canonical short name — the exact string used in every line of dialogue
  attributed to them. Titles ("Dr.", "Captain", "Elder") and surnames go in the character bio,
  NEVER in the speaker field. If you introduce "Dr. Amara Okonkwo", her canonical name is
  "Amara" and every dialogue line she speaks is attributed to "Amara" — never "Dr. Amara",
  "Okonkwo", or "Dr. Okonkwo".

**REQUIRED — Character Roster block (must appear verbatim, exactly once, at the end of your
output). Every named character in the story MUST have a line here.**
```roster
- canonical: <ShortName>  | full: <Full Name with title if any>  | expressions: neutral, worried, angry, determined
- canonical: <ShortName>  | full: <Full Name>                    | expressions: neutral, smug, thoughtful, scared
```
The `canonical` field is what will appear in every dialogue "speaker" field throughout the
story. Pick 4-8 expressions per character based on their emotional range. Valid expression
ids are ONLY: neutral, happy, sad, angry, surprised, worried, determined, smug, scared,
thoughtful. Do not invent new ones.
"""

OUTLINE_PROMPT = """
Using the World Bible provided below, create a high-level outline for the entire story.
**Requirements:**
- {target_length} total
- Each chapter should have: Chapter Number + Title, 1-2 paragraph summary, Key plot points, Major choices, Emotional tone.
- Plan for 3-5 different endings.
{idea_reminder}
{reference_reminder}

**World Bible:**\n{world_bible}
"""

CHAPTER_PROMPT = """
You are writing Chapter {chapter_number} of the Visual Novel.

**World Bible:**\n{world_bible}
**Overall Outline:**\n{outline}
**Previous Chapters Summary:**\n{previous_summary}
**Character Roster (use these EXACT speaker names — see World Bible):**\n{roster}

Write Chapter {chapter_number}. Generate 18-25 scenes.

**DIALOGUE-HEAVY PACING (very important — this is a Visual Novel, not a short story):**
- Target ratio inside "sequence": ~70% dialogue blocks, ~30% narrative blocks.
- No scene should have more than 2 narrative blocks in a row without a dialogue block breaking it up.
- Prefer short, punchy dialogue exchanges between multiple characters over long internal monologues.
- Narrative blocks are for scene-setting and physical action beats ONLY — not for restating what
  a character just said or explaining feelings the dialogue already showed. Trust the dialogue.
- A scene with zero dialogue is a code smell — if a scene has no character speaking, ask whether
  it should be merged with an adjacent scene instead.

**CHOICE DENSITY:**
- Give the player a meaningful choice every 2-4 scenes. Aim for 6-9 choice points across the chapter.
- Choices MUST diverge into different next_scene paths (not two paths that reconverge in one scene).
- For scenes with NO choices, use "next_scene_default": "next_scene_id".

**CHOICE FORMATTING RULES:**
1. Every scene with "choices" MUST include a "choice_prompt" field: 1-2 sentences of real in-world text
   (a character's question, a beat of tension, what the protagonist is weighing). NEVER use filler like
   "Make a decision" or "What will you do".
2. Each "text" in choices MUST be under 7 words — a punchy action or phrase, not a full sentence.
   Good: "Fight the guard", "Ask about the ring", "Stay silent"
   Bad: "You decide to attack the guard before he can call for backup"

**SPEAKER NAME RULE (critical for asset matching):**
- Every dialogue block's "speaker" field MUST match a canonical name from the Character Roster above,
  EXACTLY as written there. Do NOT add titles ("Dr. Amara"), surnames ("Amara Okonkwo"), or
  nicknames ("Am"). If the roster says "Amara", every line she says uses "Amara" — no exceptions.
- If a character speaks who is NOT in the roster, use a generic descriptor as the speaker:
  "Guard", "Shopkeeper", "Old Woman", "Radio Voice". Never invent a new proper name mid-chapter —
  that would create a duplicate character with no portrait asset.

**EXPRESSIONS (for character portraits — every dialogue block needs one):**
- Every "dialogue" block MUST include an "expression" field. Pick ONE of exactly these ten values:
  neutral, happy, sad, angry, surprised, worried, determined, smug, scared, thoughtful.
- Match expression to the line's emotional content. Default to "neutral" only for flat/matter-of-fact lines.
- Prefer the expressions listed for that character in the roster, but any of the ten values is valid.
- "narrative" blocks do NOT get an expression field.

**CRITICAL JSON RULES:**
1. Output ONLY valid JSON. No conversational text before or after.
2. Escape inner quotes: "text": "She said, \\"Hello.\\""
3. No trailing commas.
4. The FINAL scene of the chapter MUST NOT have choices — end linearly with "next_scene_default".

**Output ONLY valid JSON** in this exact structure:
{{
  "chapter_number": {chapter_number},
  "chapter_title": "...",
  "scenes": [
    {{
      "id": "ch{chapter_number}_scene01",
      "background": "clinic_night",
      "next_scene_default": "ch{chapter_number}_scene02",
      "sequence": [
        {{ "type": "narrative", "text": "..." }},
        {{ "type": "dialogue", "speaker": "Amara", "expression": "worried", "text": "..." }},
        {{ "type": "dialogue", "speaker": "Bayo",  "expression": "angry",   "text": "..." }}
      ],
      "choice_prompt": "The guard's hand moves to his sword. There's no more time to think.",
      "choices": [
        {{ "text": "Fight the guard", "next_scene": "ch{chapter_number}_scene02a" }},
        {{ "text": "Try to talk him down", "next_scene": "ch{chapter_number}_scene02b" }}
      ]
    }}
  ]
}}

Scenes with NO choices should omit "choice_prompt" and "choices" and just use "next_scene_default".
"""

ASSET_MANIFEST_PROMPT = """
You are cataloging the visual assets needed for a Visual Novel.

**World Bible:**
{world_bible}

**Speaker → expressions actually used in the finished story (one entry per canonical character):**
{speaker_expressions}

**Background location IDs that actually appear (one entry each, no more/fewer):**
{backgrounds}

For each character, write ONE base physical/costume description (1-2 sentences, ignoring mood —
this is what the artist reuses across every expression variant). Then, for each expression
listed for that character, write a SHORT phrase describing face/posture only (e.g. "brows drawn,
jaw tight" for angry). The artist will layer the expression on top of the base description.

For each background ID, write a concise (1-2 sentence) setting description.
Also write ONE cover art description: a striking, poster-style scene (1-2 sentences) that
captures the story's tone and would work as a thumbnail/cover image.

Output ONLY valid JSON in this exact structure:
{{
  "characters": [
    {{
      "name": "Amara",
      "base_description": "Tall Yoruba woman in a bloodstained field medic's coat, close-cropped hair, silver ear cuff.",
      "expressions": [
        {{ "id": "neutral", "note": "level gaze, mouth relaxed" }},
        {{ "id": "worried", "note": "brows drawn, lips pressed thin" }},
        {{ "id": "angry",   "note": "jaw set, eyes narrowed, chin forward" }}
      ]
    }}
  ],
  "backgrounds": [ {{ "id": "clinic_night", "description": "..." }} ],
  "cover": {{ "description": "..." }}
}}
"""

JUDGE_PROMPT = """
You are the Vystoria Quality Judge, an expert Visual Novel critic and structural editor.
Evaluate the COMPLETE generated story below across five parameters. Be strict and specific —
generic praise is not useful feedback. Cite actual scene IDs, character names, or choice text
whenever you point something out.

**World Bible:**
{world_bible}

**Full Story JSON (all chapters/scenes):**
{story_json}

**PARAMETERS (score each 1-10):**

A. Choice Impact & Player Agency — Do branching choices lead to genuinely different scenes
   (not a reworded funnel back to the same text)? Does "choice_prompt" reflect real tension?
   Are choice "text" values punchy and under ~7 words?

B. World-Bible & Lore Consistency — Does the story respect the rules, characters, and settings
   established in the World Bible? Flag any location/power/character inconsistency. Also flag
   any dialogue "speaker" value that doesn't match a canonical roster name.

C. Stylistic & Tonal Cohesion — Does the prose match the requested genre/tone and the World
   Bible's writing style guidelines? Flag generic tropes or tone-breaking modern slang.

D. Character Voice & Agency — Do characters have distinct speech patterns matching their
   profiles? Does the protagonist make active decisions rather than passively drifting?

E. Interactive UX & Narrative Flow — Is the dialogue-to-narration ratio healthy (target ~70%
   dialogue)? Are expression tags being used and do they match the emotional content of the
   lines? Flag jarring scene transitions or weak/generic choice prompts.

**CRITICAL JSON RULES:**
1. Output ONLY valid JSON. No conversational text before or after.
2. Escape inner quotes. No trailing commas.

Output ONLY valid JSON in this exact structure:
{{
  "evaluation": {{
    "overall_score": 8.1,
    "status": "PASS",
    "summary": "1-3 sentence overall verdict.",
    "metrics": {{
      "choice_impact": {{ "score": 8.5, "feedback": "..." }},
      "lore_consistency": {{ "score": 9.0, "feedback": "..." }},
      "tonal_cohesion": {{ "score": 7.2, "feedback": "..." }},
      "character_voice": {{ "score": 8.0, "feedback": "..." }},
      "narrative_flow": {{ "score": 7.5, "feedback": "..." }}
    }},
    "actionable_critiques": [
      "Specific, actionable note referencing a scene id, character, or choice."
    ]
  }}
}}
"""

# Targeted single-scene rewrite prompt. Deliberately scoped so the model
# cannot rewrite the whole book when the creator only dislikes one moment.
TWEAK_PROMPT = """
You are revising a SINGLE scene of an already-written Visual Novel, per the creator's instruction.
You are NOT rewriting the story — only this one scene. Do not reference or invent other scenes.

**Hard constraints:**
1. Preserve the scene's "id" field EXACTLY as given below. Never change it.
2. Any "next_scene" (inside choices) or "next_scene_default" value in your output MUST be copied
   verbatim from the original scene JSON below. Do NOT invent new scene ids, and do NOT remove a
   linking field that was present in the original — the rest of the story links to this scene by
   those exact ids and a dangling/renamed id will break the game.
3. You MAY change: background, sequence (narrative/dialogue text/expression), choice_prompt, and
   the wording of each choice's "text" — but if choices exist, the NUMBER of choices and their
   "next_scene" targets must stay the same as the original unless the creator's instruction
   explicitly asks you to add or remove a branch.
4. Keep the same formatting rules as the rest of the story:
   - choice_prompt required whenever choices exist (1-2 sentences of real in-world setup, never generic filler)
   - each choice "text" is a punchy phrase under ~7 words
   - every dialogue block keeps an "expression" field (one of: neutral, happy, sad, angry,
     surprised, worried, determined, smug, scared, thoughtful)
   - dialogue "speaker" values stay as the exact canonical names used in the original scene —
     don't add titles/surnames or rename anyone.

**World Bible (tone/consistency reference only):**
{world_bible}

**Original Scene JSON:**
{scene_json}

**Creator's Instruction (apply ONLY this change):**
{instruction}

**CRITICAL JSON RULES:**
1. Output ONLY valid JSON — the single revised scene object. No wrapper key, no conversational text.
2. Escape inner quotes. No trailing commas.

Output ONLY the revised scene JSON object, in the same shape as the original.
"""

JUDGE_RUBRIC = {
    "choice_impact":    {"weight": 0.30, "min_pass": 7.0},
    "lore_consistency": {"weight": 0.20, "min_pass": 8.0},
    "tonal_cohesion":   {"weight": 0.20, "min_pass": 7.0},
    "character_voice":  {"weight": 0.15, "min_pass": 7.0},
    "narrative_flow":   {"weight": 0.15, "min_pass": 6.0},
}

# Whitelist of expression ids we allow the model to use. Anything outside
# this set gets normalized to "neutral" so the frontend never has to guess
# whether "furious" or "irate" should map onto the "angry" asset slot.
VALID_EXPRESSIONS = {
    "neutral", "happy", "sad", "angry", "surprised",
    "worried", "determined", "smug", "scared", "thoughtful",
}

# ==========================================
# 4. MULTI-MODEL ADAPTER & PARSER
# ==========================================
def _is_model_unavailable(exc: Exception) -> bool:
    """True if this exception smells like 'the model name itself is bad',
    as opposed to a transient network hiccup or a payload problem.

    Checks three signals so we catch this class of failure regardless of which
    provider raised it or how their SDK phrases the error:
      1. Substring match on the error text (widest net, may need updating)
      2. Exception CLASS name (survives SDK message-format changes)
      3. HTTP status code, if the exception exposes one (most reliable when present)
    """
    if exc is None:
        return False

    # Signal 3: HTTP status code. Most SDK exceptions expose one of these
    # attributes. 404 = model doesn't exist. 400 = model rejected by vendor.
    # 403 = your account isn't allowed to use this model.
    for attr in ("status_code", "code", "http_status"):
        code = getattr(exc, attr, None)
        # Some SDKs put an object here (e.g. grpc StatusCode); coerce to str
        if code is not None and str(code) in ("404", "400", "403"):
            return True

    # Signal 2: exception class name. Works even if str(e) is empty or wrapped.
    exc_type_name = type(exc).__name__
    if exc_type_name in MODEL_UNAVAILABLE_EXCEPTION_TYPES:
        return True

    # Signal 1: substring match on the error message (case-insensitive).
    low = str(exc).lower() if exc else ""
    return any(marker in low for marker in MODEL_UNAVAILABLE_MARKERS)


def call_llm(prompt, system_instruction, provider, api_key, model_name):
    """Dynamically routes the prompt to the selected LLM provider.

    All vendor-specific SDK exceptions are caught and re-raised as either
    ModelUnavailableError (creator needs to change the model name) or plain
    Exception (transient / retryable). Both carry a message safe to show
    the creator directly."""
    provider = (provider or "").lower()
    resolved_model = (model_name or DEFAULT_MODELS.get(provider) or "").strip()
    if not resolved_model:
        raise ModelUnavailableError(
            f"No model name provided for provider '{provider}'. "
            f"Try one of: {suggested_alternatives(provider)}."
        )

    try:
        if provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(resolved_model, system_instruction=system_instruction)
            response = model.generate_content(prompt)

            if not response.candidates:
                raise Exception(
                    f"Gemini returned no candidates. prompt_feedback={getattr(response, 'prompt_feedback', None)}"
                )
            candidate = response.candidates[0]
            finish_reason = getattr(candidate, 'finish_reason', None)
            if finish_reason is not None and str(finish_reason) not in ('1', 'STOP', 'FinishReason.STOP'):
                raise Exception(
                    f"Gemini stopped generating early (finish_reason={finish_reason}). "
                    f"This usually means the safety filters blocked the content (common with "
                    f"dark/violent genres) or max_output_tokens was too low. "
                    f"safety_ratings={getattr(candidate, 'safety_ratings', None)}"
                )
            return response.text

        elif provider == 'openai':
            import openai
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content

        elif provider == 'grok':
            import openai
            client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            resp = client.chat.completions.create(
                model=resolved_model,
                messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content

        elif provider == 'claude':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            resp = client.messages.create(
                model=resolved_model,
                max_tokens=4000,
                system=system_instruction,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.content[0].text

        else:
            raise ValueError(f"Unsupported Provider: {provider}")

    except ModelUnavailableError:
        raise
    except Exception as e:
        err_str = str(e)
        # Log the exception class and any status code, so if this ever misses
        # a real "model is bad" case we can see exactly what markers to add.
        # Shows up in Render/uvicorn logs; safe to leave on in production.
        status_hint = next(
            (str(getattr(e, a)) for a in ("status_code", "code", "http_status") if getattr(e, a, None) is not None),
            "no-status"
        )
        print(f"[call_llm] {provider}/{resolved_model} raised {type(e).__name__} "
              f"(status={status_hint}): {err_str[:200]}")

        # The specific class of failure the mentor hit: vendor says the model
        # itself is not available to this account. Repromoting this into a
        # dedicated exception lets the pipeline give up with a helpful message
        # instead of retrying three times on a name that will never resolve.
        if _is_model_unavailable(e):
            first_line = err_str.splitlines()[0][:250] if err_str else "(no detail)"
            raise ModelUnavailableError(
                f"The {provider.title()} API rejected the model name "
                f"'{resolved_model}': {first_line}\n\n"
                f"👉 This usually means the model has been deprecated for new "
                f"accounts, or the model ID has a typo. Go to Engine Config and "
                f"try one of these current model IDs instead:\n"
                f"   {suggested_alternatives(provider)}"
            ) from e
        raise

def clean_json_output(raw_text):
    """Zero-Regex Brace Counting Algorithm."""
    start_idx = raw_text.find('{')
    if start_idx != -1:
        brace_count = 0
        for i in range(start_idx, len(raw_text)):
            if raw_text[i] == '{':
                brace_count += 1
            elif raw_text[i] == '}':
                brace_count -= 1
            if brace_count == 0:
                return raw_text[start_idx:i+1]

        end_idx = raw_text.rfind('}')
        if end_idx != -1 and end_idx > start_idx:
            return raw_text[start_idx:end_idx+1]
    return raw_text.strip()


def parse_character_roster(world_bible: str):
    """Extracts the ```roster fenced block from the World Bible.
    Returns (roster_text_for_prompt, canonical_names_set, expressions_by_char)."""
    m = re.search(r"```roster\s*(.*?)```", world_bible, re.DOTALL | re.IGNORECASE)
    if not m:
        return "(no roster provided — use single-name speakers only)", set(), {}

    canonical_names = set()
    expressions_by_char = {}
    lines_for_prompt = []

    for line in m.group(1).splitlines():
        line = line.strip().lstrip("-").strip()
        if not line:
            continue
        parts = {p.split(":", 1)[0].strip().lower(): p.split(":", 1)[1].strip()
                 for p in line.split("|") if ":" in p}
        canon = parts.get("canonical")
        if not canon:
            continue
        canonical_names.add(canon)
        exprs = [e.strip() for e in parts.get("expressions", "neutral").split(",") if e.strip()]
        expressions_by_char[canon] = exprs
        lines_for_prompt.append(f"- {canon} ({parts.get('full', canon)}) — expressions: {', '.join(exprs)}")

    return "\n".join(lines_for_prompt) or "(roster block was empty)", canonical_names, expressions_by_char


def normalize_speakers(scenes, canonical_names):
    """Rewrites any dialogue speaker that fuzzy-matches a canonical name to that canonical name.
    Also normalizes each dialogue block's "expression" field."""
    canon_by_lower = {c.lower(): c for c in canonical_names} if canonical_names else {}

    for scene in scenes:
        for block in scene.get("sequence", []):
            if block.get("type") != "dialogue":
                continue

            spoken = (block.get("speaker") or "").strip()
            if spoken and canonical_names and spoken not in canonical_names:
                spoken_l = spoken.lower()
                match = canon_by_lower.get(spoken_l)
                if not match:
                    for lower_c, canon in canon_by_lower.items():
                        if re.search(rf"\b{re.escape(lower_c)}\b", spoken_l):
                            match = canon
                            break
                if match:
                    block["speaker"] = match

            expr = (block.get("expression") or "").strip().lower()
            if expr not in VALID_EXPRESSIONS:
                block["expression"] = "neutral"
            else:
                block["expression"] = expr

    return scenes


def build_asset_manifest(world_bible, speaker_expressions_map, background_ids, provider, api_key, model_name):
    """Catalogs character/background/cover art descriptions with per-expression variants.
    Never raises — falls back to a bare-name manifest if the LLM call or JSON parse fails."""
    speaker_expressions_text = "\n".join(
        f"- {name}: {', '.join(sorted(exprs)) or 'neutral'}"
        for name, exprs in sorted(speaker_expressions_map.items())
    ) or "(none)"

    try:
        manifest_raw = call_llm(
            ASSET_MANIFEST_PROMPT.format(
                world_bible=world_bible,
                speaker_expressions=speaker_expressions_text,
                backgrounds="\n".join(f"- {b}" for b in background_ids) or "(none)"
            ),
            "Output ONLY valid JSON.", provider, api_key, model_name
        )
        asset_manifest = json.loads(clean_json_output(manifest_raw))

        described_chars = {c.get('name'): c for c in asset_manifest.get('characters', [])}
        for name, used_exprs in speaker_expressions_map.items():
            char = described_chars.get(name)
            if not char:
                asset_manifest.setdefault('characters', []).append({
                    "name": name,
                    "base_description": "",
                    "expressions": [{"id": e, "note": ""} for e in sorted(used_exprs)]
                })
                continue
            char.setdefault("base_description", char.pop("description", "") or "")
            existing = {e.get('id') for e in char.get('expressions', [])}
            for e in used_exprs:
                if e not in existing:
                    char.setdefault('expressions', []).append({"id": e, "note": ""})

        described_bgs = {b.get('id') for b in asset_manifest.get('backgrounds', [])}
        for bg in background_ids:
            if bg not in described_bgs:
                asset_manifest.setdefault('backgrounds', []).append({"id": bg, "description": ""})

        asset_manifest.setdefault("cover", {"description": ""})
        return asset_manifest, None

    except Exception as e:
        fallback = {
            "characters": [
                {
                    "name": name,
                    "base_description": "",
                    "expressions": [{"id": ex, "note": ""} for ex in sorted(exprs or {"neutral"})]
                }
                for name, exprs in sorted(speaker_expressions_map.items())
            ],
            "backgrounds": [{"id": b, "description": ""} for b in background_ids],
            "cover": {"description": ""}
        }
        return fallback, str(e)


def run_judge_evaluation(world_bible, final_story, provider, api_key, model_name):
    """Runs the LLM-as-a-Judge QA stage and returns a scorecard dict. Never raises."""
    try:
        raw = call_llm(
            JUDGE_PROMPT.format(
                world_bible=world_bible,
                story_json=json.dumps(final_story, ensure_ascii=False)
            ),
            "You are a rigorous, detail-oriented Visual Novel quality judge. Output ONLY valid JSON.",
            provider, api_key, model_name
        )
        parsed = json.loads(clean_json_output(raw))
        evaluation = parsed.get("evaluation", parsed)
        metrics = evaluation.get("metrics", {})

        weighted_sum = 0.0
        total_weight = 0.0
        failed_params = []
        for key, rule in JUDGE_RUBRIC.items():
            entry = metrics.get(key, {})
            score = entry.get("score")
            if not isinstance(score, (int, float)):
                continue
            weighted_sum += score * rule["weight"]
            total_weight += rule["weight"]
            if score < rule["min_pass"]:
                failed_params.append(key)

        overall_score = round(weighted_sum / total_weight, 2) if total_weight else None
        status = "FAIL" if (overall_score is None or overall_score < 7.5 or failed_params) else "PASS"

        return {
            "overall_score": overall_score,
            "status": status,
            "failed_parameters": failed_params,
            "summary": evaluation.get("summary", ""),
            "metrics": metrics,
            "actionable_critiques": evaluation.get("actionable_critiques", []),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }, None
    except Exception as e:
        return {
            "overall_score": None,
            "status": "ERROR",
            "failed_parameters": [],
            "summary": f"AI evaluation could not be completed automatically: {e}",
            "metrics": {},
            "actionable_critiques": [],
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }, str(e)


def tweak_scene(world_bible, scene, instruction, provider, api_key, model_name):
    """Rewrites ONE scene per a targeted creator instruction."""
    original_id = scene.get("id")
    raw = call_llm(
        TWEAK_PROMPT.format(
            world_bible=world_bible or "(none provided)",
            scene_json=json.dumps(scene, ensure_ascii=False),
            instruction=instruction
        ),
        "Output ONLY valid JSON for the single revised scene. Never change 'id' or invent new scene ids.",
        provider, api_key, model_name
    )
    revised = json.loads(clean_json_output(raw))

    revised["id"] = original_id

    original_targets = set()
    for c in (scene.get("choices") or []):
        if c.get("next_scene"):
            original_targets.add(c["next_scene"])
    if scene.get("next_scene_default"):
        original_targets.add(scene["next_scene_default"])

    if revised.get("choices"):
        for c in revised["choices"]:
            if c.get("next_scene") and original_targets and c["next_scene"] not in original_targets:
                raise Exception(
                    f"Model invented a new next_scene id ('{c['next_scene']}') that wasn't in the "
                    f"original scene. Try a more specific instruction (e.g. don't ask it to add a "
                    f"new branch unless you also want to wire it up manually)."
                )
    if revised.get("next_scene_default") and original_targets and \
       revised["next_scene_default"] not in original_targets:
        raise Exception(
            f"Model invented a new next_scene_default ('{revised['next_scene_default']}') that wasn't "
            f"in the original scene."
        )

    normalize_speakers([revised], set())

    return revised

# ==========================================
# 5. ASYNC BACKGROUND WORKER THREAD
# ==========================================
def run_generation_pipeline(task_id: str, req: GenerateRequest):
    logs = []

    def update_task(status, current_step, progress, log_msg=None, final_url=None):
        if log_msg:
            print(f"[{task_id}] {log_msg}")
            logs.append(log_msg)

        payload = {
            "status": status,
            "current_step": current_step,
            "progress_percent": progress,
            "logs": logs,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        if final_url:
            payload["final_url"] = final_url

        try:
            supabase.table("generation_tasks").update(payload).eq("id", task_id).execute()
        except Exception as db_err:
            print(f"[{task_id}] ⚠️ Failed to write task update to Supabase: {db_err}")
            traceback.print_exc()

    try:
        num_chapters = 8
        match = re.search(r'\d+', req.target_length)
        if match:
            num_chapters = int(match.group())

        naming_pool = pick_naming_pool(3)

        update_task('generating', 'Building World Bible...', 5,
                    f"Started {req.provider.upper()} engine using model '{req.model_name}'. "
                    f"Chapters targeted: {num_chapters}. Naming pool for this generation: {naming_pool}.")

        idea_section = (
            f"- Core Idea (build the plot and world firmly around this creator-provided concept): {req.idea}"
            if req.idea and req.idea.strip() else ""
        )
        idea_reminder = (
            f"- Stay faithful to this core idea from the creator: {req.idea}"
            if req.idea and req.idea.strip() else ""
        )

        has_reference = bool(req.reference_text and req.reference_text.strip())
        trimmed_reference = (req.reference_text or "").strip()[:MAX_REFERENCE_CHARS]
        reference_section = (
            f"- A Reference Document has been provided below. Treat it as the AUTHORITATIVE source: "
            f"adapt its plot, characters, and setting faithfully into the World Bible format rather "
            f"than inventing a different story. Only invent details necessary to fill gaps (minor "
            f"side characters, extra locations) while staying fully consistent with the document. "
            f"If the reference already names characters, keep those names AS-IS in the roster (the "
            f"naming-pool rule above only applies to characters you invent to fill gaps).\n\n"
            f"**Reference Document:**\n{trimmed_reference}\n"
            if has_reference else ""
        )
        reference_reminder = (
            "- This outline MUST follow the plot/structure of the Reference Document supplied when "
            "building the World Bible — do not diverge from it."
            if has_reference else ""
        )
        if has_reference:
            update_task('generating', 'Building World Bible...', 5,
                        f"Reference document detected ({len(trimmed_reference)} chars) — adapting it instead of freeform generation.")

        world_bible = call_llm(
            WORLD_PROMPT.format(title=req.title, subtitle=req.subtitle, genre=req.genre,
                                 target_length=req.target_length, tone=req.tone,
                                 idea_section=idea_section, reference_section=reference_section,
                                 naming_pool=naming_pool),
            "You are a master visual novel author.", req.provider, req.api_key, req.model_name
        )

        roster_prompt, canonical_names, expressions_by_char = parse_character_roster(world_bible)
        if canonical_names:
            update_task('generating', 'World Bible ready.', 15,
                        f"Roster locked in with {len(canonical_names)} canonical character(s): "
                        f"{', '.join(sorted(canonical_names))}.")
        else:
            update_task('generating', 'World Bible ready.', 15,
                        "⚠️ No roster block found in World Bible — speaker names won't be normalized. "
                        "Consider regenerating if you see duplicate characters like 'Amara' and 'Dr. Amara'.")

        outline = call_llm(
            OUTLINE_PROMPT.format(target_length=req.target_length, world_bible=world_bible,
                                   idea_reminder=idea_reminder, reference_reminder=reference_reminder),
            "You are a master visual novel author.", req.provider, req.api_key, req.model_name
        )

        update_task('generating', 'Writing Chapters...', 25, "Master outline locked in. Beginning chapter pipeline.")

        all_scenes = []
        starting_scene = None
        prev_last_scene = None
        previous_summary = "This is the very beginning."
        MAX_RETRIES = 3

        for i in range(1, num_chapters + 1):
            step_msg = f"Writing Chapter {i} of {num_chapters}..."
            base_prog = 25 + int((i / num_chapters) * 60)
            update_task('generating', step_msg, base_prog, step_msg)

            prompt = CHAPTER_PROMPT.format(
                chapter_number=i,
                world_bible=world_bible,
                outline=outline,
                previous_summary=previous_summary,
                roster=roster_prompt,
            )

            scenes = []
            for attempt in range(MAX_RETRIES):
                try:
                    raw_data = call_llm(prompt, "Output ONLY valid JSON. You MUST escape inner quotes like \\\"this\\\".", req.provider, req.api_key, req.model_name)
                    chapter_data = json.loads(clean_json_output(raw_data))
                    scenes = chapter_data.get("scenes", [])
                    if scenes:
                        update_task('generating', step_msg, base_prog, f"Chapter {i} structured and validated.")
                        break
                except ModelUnavailableError:
                    # Model itself is bad — retrying won't help, bail immediately.
                    raise
                except Exception as e:
                    update_task('generating', step_msg, base_prog, f"⚠️ Attempt {attempt + 1} Failed (JSON error). Retrying...")
                    time.sleep(2)

            if not scenes:
                raise Exception(f"Failed to generate valid JSON for Chapter {i} after {MAX_RETRIES} attempts.")

            scenes = normalize_speakers(scenes, canonical_names)

            if prev_last_scene:
                target_scene_id = scenes[0]["id"]
                if prev_last_scene.get("choices"):
                    for choice in prev_last_scene["choices"]:
                        choice["next_scene"] = target_scene_id
                else:
                    prev_last_scene["next_scene_default"] = target_scene_id

            all_scenes.extend(scenes)
            prev_last_scene = scenes[-1]
            previous_summary += f"\nChapter {i} completed."

            if not starting_scene and i == 1:
                starting_scene = scenes[0]["id"]

            if i < num_chapters:
                time.sleep(3)

        speaker_expressions_map: dict[str, set[str]] = {}
        for scene in all_scenes:
            for block in scene.get("sequence", []):
                if block.get("type") == "dialogue" and block.get("speaker"):
                    spk = block["speaker"]
                    expr = block.get("expression") or "neutral"
                    speaker_expressions_map.setdefault(spk, set()).add(expr)

        background_ids = sorted({
            scene.get('background') for scene in all_scenes if scene.get('background')
        })

        final_title = f"{req.title}: {req.subtitle}"
        final_story = {
            "title": final_title,
            "starting_scene": starting_scene or "ch1_scene01",
            "scenes": all_scenes
        }

        update_task('generating', 'Cataloging assets & running AI Judge...', 88,
                    f"Found {len(speaker_expressions_map)} unique speakers across "
                    f"{sum(len(v) for v in speaker_expressions_map.values())} portrait variants. "
                    "Starting asset manifest and AI Judge in parallel...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            assets_future = executor.submit(
                build_asset_manifest, world_bible, speaker_expressions_map, background_ids,
                req.provider, req.api_key, req.model_name
            )
            judge_future = executor.submit(
                run_judge_evaluation, world_bible, final_story,
                req.provider, req.api_key, req.model_name
            )
            asset_manifest, asset_err = assets_future.result()
            evaluation_scorecard, judge_err = judge_future.result()

        if asset_err:
            update_task('generating', 'Asset manifest ready (fallback).', 92,
                        f"⚠️ Could not auto-describe assets ({asset_err}); showing names/expressions only.")
        else:
            total_variants = sum(len(c.get('expressions', [])) for c in asset_manifest.get('characters', []))
            update_task('generating', 'Asset manifest ready.', 92,
                        f"Cataloged {len(asset_manifest.get('characters', []))} characters "
                        f"({total_variants} portrait variants) and {len(background_ids)} backgrounds.")

        if judge_err:
            update_task('generating', 'AI evaluation skipped.', 95,
                        f"⚠️ AI Judge could not complete automatically ({judge_err}). Manual review still required.")
        else:
            score = evaluation_scorecard.get('overall_score')
            update_task('generating', 'AI evaluation complete.', 95,
                        f"🧑‍⚖️ Judge verdict: {evaluation_scorecard['status']}"
                        + (f" (Weighted Score: {score}/10)" if score is not None else "")
                        + ". This is advisory only — review the scorecard and decide for yourself.")

        try:
            supabase.table("generation_tasks").update({
                "result_json": final_story,
                "asset_manifest": asset_manifest,
            }).eq("id", task_id).execute()
        except Exception as db_err:
            traceback.print_exc()
            update_task('failed', 'Error saving story', 96,
                        f"❌ Failed to save the story to Supabase: {db_err}")
            return

        try:
            supabase.table("generation_tasks").update({
                "evaluation_scorecard": evaluation_scorecard,
                "world_bible": world_bible,
            }).eq("id", task_id).execute()
        except Exception as db_err:
            traceback.print_exc()
            update_task('generating', 'Evaluation not saved', 97,
                        f"⚠️ Story saved fine, but couldn't save the AI Judge scorecard "
                        f"(have you run supabase_migration_add_evaluation.sql?): {db_err}")

        update_task('completed', 'Ready for review', 100,
                    "✅ Story generated! Play-test it end-to-end to unlock draft saving.")

    except ModelUnavailableError as mue:
        # Distinct from a generic FATAL ERROR — this one is fully actionable
        # by the creator (change the model name), so we say so plainly.
        traceback.print_exc()
        update_task('failed', 'Model unavailable', 0, f"❌ {str(mue)}")

    except Exception as e:
        traceback.print_exc()
        update_task('failed', 'Error occurred', 0, f"❌ FATAL ERROR: {str(e)}")

# ==========================================
# 6. API ENDPOINTS
# ==========================================
@app.get("/")
def health_check():
    return {"status": "Vystoria Multi-Model Server is running!"}

@app.post("/generate")
def generate_story_endpoint(req: GenerateRequest, background_tasks: BackgroundTasks):
    print(f"🚀 Received Request: {req.provider} - {req.title}")

    try:
        res = supabase.table("generation_tasks").insert({
            "creator_id": req.user_id,
            "title": f"{req.title}: {req.subtitle}",
            "provider": req.provider,
            "status": "pending",
            "current_step": "Initializing..."
        }).execute()

        if not res.data:
            raise Exception("Insert returned no data — check that the 'generation_tasks' table exists and RLS policies allow this insert.")

        task_id = res.data[0]["id"]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create generation task: {str(e)}")

    background_tasks.add_task(run_generation_pipeline, task_id, req)

    return {
        "status": "success",
        "message": "Generation task started in the background.",
        "task_id": task_id
    }

@app.post("/evaluate/{task_id}")
def evaluate_story_endpoint(task_id: str, req: EvaluateRequest):
    try:
        res = supabase.table("generation_tasks").select("*").eq("id", task_id).single().execute()
        row = res.data
        if not row or not row.get("result_json"):
            raise HTTPException(status_code=404, detail="No generated story found for this task_id yet.")

        world_bible = row.get("world_bible") or ""
        final_story = row["result_json"]

        evaluation_scorecard, judge_err = run_judge_evaluation(
            world_bible, final_story, req.provider, req.api_key, req.model_name
        )

        supabase.table("generation_tasks").update({
            "evaluation_scorecard": evaluation_scorecard
        }).eq("id", task_id).execute()

        return {"status": "success", "evaluation_scorecard": evaluation_scorecard, "error": judge_err}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

@app.post("/tweak-scene")
def tweak_scene_endpoint(req: TweakSceneRequest):
    """Rewrites exactly one scene per the creator's instruction."""
    if not req.scene or not req.scene.get("id"):
        raise HTTPException(status_code=400, detail="Scene payload is missing an 'id'.")
    if not req.instruction or not req.instruction.strip():
        raise HTTPException(status_code=400, detail="Instruction cannot be empty.")

    try:
        revised_scene = tweak_scene(
            req.world_bible, req.scene, req.instruction.strip(),
            req.provider, req.api_key, req.model_name
        )
        return {"status": "success", "scene": revised_scene}
    except ModelUnavailableError as mue:
        raise HTTPException(status_code=400, detail=str(mue))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scene tweak failed: {str(e)}")