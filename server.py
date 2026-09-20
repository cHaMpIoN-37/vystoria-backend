import os
import json
import re
import time
import base64
import random
import hmac
import hashlib
import secrets
import threading
import httpx
import traceback
from datetime import datetime, timezone, timedelta
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
# months as providers retire models, so keep them fresh. The frontend sends a
# user-editable model string; these are just safety nets.
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",     # stable, replaces the 2.5-flash line
    "openai": "gpt-4o",
    "claude": "claude-sonnet-4-6",    # stable Sonnet 4 tier — broadest availability
    "grok":   "grok-2-latest",
}

# Image models are a SEPARATE provider+key from the text engine, so a creator
# can write with (say) OpenAI and draw with Gemini without one eating the
# other's daily quota.
DEFAULT_IMAGE_MODELS = {
    "gemini": "gemini-2.5-flash-image",
    "openai": "gpt-image-1",
}

# Output-token ceilings. THE SINGLE BIGGEST SOURCE OF WASTED QUOTA in the old
# code: no ceiling was ever passed, so a 20-scene chapter would silently hit
# the provider's default output cap, come back as truncated JSON, and burn
# three retries producing the same truncation every time.
MAX_OUTPUT_TOKENS = {
    "gemini": int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "32768")),
    "openai": int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "16384")),
    "claude": int(os.environ.get("CLAUDE_MAX_OUTPUT_TOKENS", "16384")),
    "grok":   int(os.environ.get("GROK_MAX_OUTPUT_TOKENS", "16384")),
}

# Minimum wall-clock spacing between two calls made with the SAME api key.
# Free-tier Gemini is roughly 10 requests/minute; 4s spacing keeps us under
# that without the creator ever seeing a 429.
MIN_SECONDS_BETWEEN_CALLS = float(os.environ.get("LLM_MIN_INTERVAL", "4"))

# Back-off policy for *transient* 429s (per-minute RPM/TPM, overload).
# Daily-quota 429s are never retried — see _classify_rate_limit().
MAX_RATE_LIMIT_RETRIES = int(os.environ.get("LLM_RATE_LIMIT_RETRIES", "4"))
RATE_LIMIT_BASE_SLEEP  = float(os.environ.get("LLM_RATE_LIMIT_BASE_SLEEP", "8"))
MAX_RATE_LIMIT_SLEEP   = float(os.environ.get("LLM_RATE_LIMIT_MAX_SLEEP", "90"))

# Per-chapter attempts. Lower than the old 3 because json_mode + an explicit
# output ceiling removes almost every reason a chapter used to fail.
CHAPTER_MAX_RETRIES = int(os.environ.get("CHAPTER_MAX_RETRIES", "2"))

# Bumping this invalidates every stored checkpoint (use when the checkpoint
# shape changes, so a resume can't half-restore an incompatible payload).
CHECKPOINT_VERSION = 2

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


class RateLimitedError(Exception):
    """A 429 that a short sleep CAN clear: per-minute request/token caps, or a
    momentarily overloaded endpoint. Carries the vendor's own suggested wait
    where one was supplied."""
    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


class QuotaExhaustedError(Exception):
    """A ceiling that retrying inside this run CANNOT clear: the free-tier
    per-day request cap, or the creator's own per-run call budget.

    This is the fix for the reported failure mode. The old code caught a bare
    Exception around every chapter and retried three times — so a daily-quota
    429 (which will not succeed again until midnight Pacific) spent THREE
    requests per chapter proving the same point, ~24 wasted requests against a
    20/day allowance. Raising a distinct type lets the pipeline stop dead,
    keep its checkpoint, and tell the creator to come back tomorrow or swap
    keys."""


class TruncatedOutputError(Exception):
    """The model ran into its output-token ceiling part-way through the JSON.
    Retrying the identical prompt reproduces it exactly, so the caller shrinks
    the requested scene count instead of retrying blind."""


def _extract_retry_after(text: str):
    """Pulls a suggested wait out of whatever shape the vendor used.
    Gemini: `retry_delay { seconds: 22 }` and `Please retry in 22.07s`."""
    for pattern in (
        r"retry_delay\s*\{\s*seconds:\s*(\d+)",
        r"retry in\s*([\d.]+)\s*s",
        r"try again in\s*([\d.]+)\s*s",
        r"retry-after[\"']?\s*[:=]\s*([\d.]+)",
    ):
        m = re.search(pattern, text or "", re.IGNORECASE)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                pass
    return None


RATE_LIMIT_MARKERS = (
    "429",
    "rate limit",
    "rate_limit",
    "ratelimit",
    "resource_exhausted",
    "resource has been exhausted",
    "too many requests",
    "exceeded your current quota",
    "quota exceeded",
    "overloaded",
    "please retry in",
)

# Substrings that mean "this is a PER-DAY ceiling, not a per-minute one".
# `GenerateRequestsPerDayPerProjectPerModel-FreeTier` lowercases to contain
# "perday", which is what catches the exact error in the bug report.
DAILY_QUOTA_MARKERS = (
    "perday",
    "per day",
    "per-day",
    "requests per day",
    "daily limit",
    "daily quota",
    "free_tier_requests",
    "generate_content_free_tier_requests",
    "check your plan and billing",
    "insufficient_quota",
)


def _classify_rate_limit(exc: Exception):
    """Returns 'daily', 'transient', or None.

    'daily'     -> QuotaExhaustedError, stop the run, keep the checkpoint.
    'transient' -> RateLimitedError, sleep and retry.
    """
    if exc is None:
        return None

    low = f"{type(exc).__name__} {exc}".lower()
    status = next(
        (str(getattr(exc, a)) for a in ("status_code", "code", "http_status")
         if getattr(exc, a, None) is not None),
        "",
    )

    looks_rate_limited = "429" in status or any(m in low for m in RATE_LIMIT_MARKERS)
    if not looks_rate_limited:
        return None
    if any(m in low for m in DAILY_QUOTA_MARKERS):
        return "daily"
    return "transient"


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

    # --- quota controls (all optional; safe defaults) ---
    # 'off'      — skip the judge entirely. Cheapest run possible.
    # 'advisory' — run it once, record the scorecard, NEVER regenerate. Default.
    # 'strict'   — the old behaviour: regenerate every chapter once on a FAIL.
    judge_mode: str = "advisory"
    # Scenes asked for per chapter. The old hardcoded "18-25" is what kept
    # overrunning the output ceiling on Flash-tier models.
    scenes_per_chapter: int = 14
    # Hard ceiling on model calls for this run. 0 = unlimited.
    max_llm_calls: int = 0


class EvaluateRequest(BaseModel):
    """Lets the creator manually (re-)trigger the AI Judge for an already-generated
    task, e.g. with a different provider/model than was used to write the story."""
    provider: str
    api_key: str
    model_name: str


class ResumeRequest(BaseModel):
    """Restarts a checkpointed task from the last completed chapter. The key
    and model may differ from the original run — that is the point: a Gemini
    run that hit the daily wall can be finished on an OpenAI key."""
    provider: str
    api_key: str
    model_name: str
    judge_mode: str = "advisory"
    scenes_per_chapter: int = 14
    max_llm_calls: int = 0


class ImageRequest(BaseModel):
    """One asset, generated on a SEPARATE provider/key from the text engine."""
    provider: str            # 'gemini' | 'openai'
    api_key: str
    model_name: str | None = None
    prompt: str
    kind: str = "character"  # character | background | cover — drives aspect ratio
    style: str | None = None # house art-style preset, prepended to every prompt
    # The creator app composes the full prompt itself (style + format +
    # composition + negatives) so that its Copy button and its Generate button
    # produce identical text. When it does, it sends compose=false and we pass
    # the prompt through untouched. Anything else calling this endpoint can
    # leave it at the default and still get the rules appended server-side.
    compose: bool = True


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

**Chapter context:** {chapter_context}

**World Bible:**\n{world_bible}
**Overall Outline:**\n{outline}
**Previous Chapters Summary:**\n{previous_summary}
**Character Roster (use these EXACT speaker names — see World Bible):**\n{roster}

Write Chapter {chapter_number}. Generate EXACTLY {scene_count} scenes — no more.
Do not pad past {scene_count}; running long overflows the output-token limit and
the whole chapter has to be regenerated.

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
You are the art director and copywriter for a Visual Novel. You are writing the
brief an illustrator will work from, plus the store blurb players will read.

**World Bible:**
{world_bible}

**Speaker → expressions actually used in the finished story (one entry per canonical character):**
{speaker_expressions}

**Background location IDs that actually appear (one entry each, no more/fewer):**
{backgrounds}

═══════════════════════════════════════════════════════════════════
WHAT TO WRITE
═══════════════════════════════════════════════════════════════════

**1. synopsis** — 2 to 3 sentences of back-cover copy, present tense, written to
make a browsing player tap. Name the protagonist, the world, and the central
tension. End on the stakes or a hook. NO spoilers past the first act, no
meta-language ("in this visual novel...", "players will..."), no ending reveals.

**2. art_direction** — ONE sentence naming the house visual style every asset
shares: medium, line quality, shading, and palette. It is prepended to every
image prompt, so it must be concrete and reusable. Anchor it to a FLAT 2D
ILLUSTRATED medium — never photorealism, never 3D rendering.
Good: "Flat 2D cel-shaded anime illustration, crisp black line art, muted
teal-and-rust palette with hard-edged shadows and a single warm key light."
Bad: "Dark and atmospheric." (Not a medium. Not reusable.)

**3. characters** — for each, ONE base description the artist reuses across
every expression variant. Cover, in this order:
  - apparent age, build, height, skin tone, hair (colour, length, how it sits)
  - full outfit head to toe, including footwear, plus fabric and wear/damage
  - one or two signature props or markings that make them instantly recognisable
  - their 2-3 colour signature palette
Describe the person and costume ONLY — never the mood, never the setting,
never the pose or camera. Those come from the expression note and the
composition rules. 2 to 3 sentences.

Then, for each expression listed for that character, a SHORT phrase covering
face and posture only — eyebrows, eyes, mouth, head tilt, shoulders. No
scenery, no lighting, no clothing.

**4. backgrounds** — for each ID, 2 sentences: the location, the time of day,
the light source and its direction, and the two or three objects that establish
the place. Write it as an EMPTY stage. No people, no characters, no animals —
the cast is composited on top at runtime.

**5. cover** — one poster-style key-art description. A single striking focal
subject dead centre, readable at thumbnail size. Do NOT describe any lettering,
title, or logo — the app draws the title itself, and baked-in text renders as
garbled glyphs.

═══════════════════════════════════════════════════════════════════
CRITICAL JSON RULES
═══════════════════════════════════════════════════════════════════
1. Output ONLY valid JSON. No conversational text before or after.
2. Escape inner quotes. No trailing commas.
3. Every character in the speaker list above gets an entry. Every background ID
   gets an entry. No extras, none missing.

Output ONLY valid JSON in this exact structure:
{{
  "synopsis": "Amara has spent six years stitching soldiers back together in a clinic the war forgot. The night a dying courier presses a sealed ledger into her hands, she learns the ceasefire she has been praying for was bought with her own village. Every choice now decides who she becomes when the truth finally surfaces.",
  "art_direction": "Flat 2D cel-shaded anime illustration, crisp black line art, muted teal-and-rust palette with hard-edged shadows and a single warm key light.",
  "characters": [
    {{
      "name": "Amara",
      "base_description": "Tall Yoruba woman in her early thirties, lean and broad-shouldered, deep brown skin, close-cropped black hair. She wears a field medic's canvas coat over a grey collarless shirt, the coat rust-stained at the cuffs and missing its second button, dark trousers tucked into scuffed leather boots. A silver ear cuff on her left ear and a worn leather satchel across her body. Palette: oxidised teal, dried rust, dull silver.",
      "expressions": [
        {{ "id": "neutral", "note": "level gaze, mouth relaxed, chin slightly lowered" }},
        {{ "id": "worried", "note": "brows drawn together, lips pressed thin, shoulders tight" }},
        {{ "id": "angry",   "note": "jaw set, eyes narrowed, chin pushed forward" }}
      ]
    }}
  ],
  "backgrounds": [
    {{ "id": "clinic_night", "description": "A cramped field clinic after dark, three cots along a sandbagged wall and a shelf of mismatched glass bottles. A single hooded oil lamp hangs low over the centre cot, throwing hard light downward and leaving the corners in deep blue shadow." }}
  ],
  "cover": {{ "description": "A lone medic standing centred in a lamplit doorway, a sealed ledger held against her chest, the dark of the war-torn street swallowing the space behind her." }}
}}
"""

JUDGE_PROMPT = """
You are the Vystoria Quality Judge, an expert Visual Novel critic and structural editor.
Evaluate the COMPLETE generated story below across five parameters. Be strict and specific —
generic praise is not useful feedback. Cite actual scene IDs, character names, or choice text
whenever you point something out.

**World Bible:**
{world_bible}

**Machine-computed structural facts (these are measured, not estimated — trust
them over your own counting, and cite them in your feedback where relevant):**
{story_stats}

**Story digest.** Every scene is listed with its links and choices. A
representative sample of scenes is shown with full dialogue and narration;
the rest are summarized as counts + speakers. Judge prose quality from the
sampled scenes and structure from the full listing.
{story_digest}

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

# Thresholds relaxed. The old bar (overall >= 7.5 AND every single metric at
# or above its own minimum, with lore_consistency needing 8.0/10) meant one
# pedantic note about a background id dropped the whole draft — and under the
# old MAX_JUDGE_ATTEMPTS=2 that cost a SECOND full chapter run, doubling the
# quota spend of the entire generation. A judge FAIL is now advisory by
# default; see `judge_mode` on GenerateRequest.
JUDGE_RUBRIC = {
    "choice_impact":    {"weight": 0.30, "min_pass": 6.5},
    "lore_consistency": {"weight": 0.20, "min_pass": 7.0},
    "tonal_cohesion":   {"weight": 0.20, "min_pass": 6.5},
    "character_voice":  {"weight": 0.15, "min_pass": 6.5},
    "narrative_flow":   {"weight": 0.15, "min_pass": 6.0},
}

JUDGE_PASS_SCORE = float(os.environ.get("JUDGE_PASS_SCORE", "7.0"))

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


def _openai_style_chat(client, resolved_model, system_instruction, prompt,
                       max_tokens, temperature, json_mode):
    """Shared by the 'openai' and 'grok' branches. Handles the two ways newer
    endpoints reject older kwargs (max_tokens -> max_completion_tokens, and
    response_format unsupported) without spending an extra generation."""
    kwargs = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        retried = False
        if "max_tokens" in msg and "max_completion_tokens" in msg:
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            retried = True
        if "response_format" in msg:
            kwargs.pop("response_format", None)
            retried = True
        if not retried:
            raise
        return client.chat.completions.create(**kwargs)


def call_llm(prompt, system_instruction, provider, api_key, model_name,
             *, json_mode=False, max_output_tokens=None, temperature=None):
    """Dynamically routes the prompt to the selected LLM provider.

    Three things changed here versus the old version, all of them quota fixes:

    1. `json_mode` turns on the provider's native structured-output switch
       (Gemini response_mime_type, OpenAI/Grok response_format, Claude
       assistant prefill). Most "⚠️ Attempt N Failed (JSON error)" retries in
       the logs were markdown fences or a chatty preamble — this removes the
       whole class, and each removed retry is a request back in the budget.

    2. An explicit output ceiling is always sent, and a MAX_TOKENS finish is
       raised as TruncatedOutputError rather than a generic Exception, so the
       caller shrinks the ask instead of retrying the identical prompt.

    3. 429s are split into RateLimitedError (sleep + retry) and
       QuotaExhaustedError (stop, keep the checkpoint, tell the creator).
    """
    provider = (provider or "").lower()
    resolved_model = (model_name or DEFAULT_MODELS.get(provider) or "").strip()
    if not resolved_model:
        raise ModelUnavailableError(
            f"No model name provided for provider '{provider}'. "
            f"Try one of: {suggested_alternatives(provider)}."
        )

    max_tokens = int(max_output_tokens or MAX_OUTPUT_TOKENS.get(provider, 8192))

    try:
        if provider == 'gemini':
            import google.generativeai as genai
            genai.configure(api_key=api_key)

            gen_config = {"max_output_tokens": max_tokens}
            if temperature is not None:
                gen_config["temperature"] = temperature
            if json_mode:
                gen_config["response_mime_type"] = "application/json"

            model = genai.GenerativeModel(
                resolved_model,
                system_instruction=system_instruction,
                generation_config=gen_config,
            )
            response = model.generate_content(prompt)

            if not response.candidates:
                raise Exception(
                    f"Gemini returned no candidates. prompt_feedback={getattr(response, 'prompt_feedback', None)}"
                )

            candidate = response.candidates[0]
            finish_reason = str(getattr(candidate, 'finish_reason', '') or '')

            if finish_reason in ('2', 'MAX_TOKENS', 'FinishReason.MAX_TOKENS'):
                raise TruncatedOutputError(
                    f"Gemini hit its {max_tokens}-token output ceiling before finishing. "
                    f"The response is incomplete JSON."
                )
            if finish_reason and finish_reason not in ('1', 'STOP', 'FinishReason.STOP'):
                raise Exception(
                    f"Gemini stopped generating early (finish_reason={finish_reason}). "
                    f"This usually means the safety filters blocked the content (common with "
                    f"dark/violent genres). "
                    f"safety_ratings={getattr(candidate, 'safety_ratings', None)}"
                )
            return response.text

        elif provider in ('openai', 'grok'):
            import openai
            if provider == 'grok':
                client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            else:
                client = openai.OpenAI(api_key=api_key)

            resp = _openai_style_chat(
                client, resolved_model, system_instruction, prompt,
                max_tokens, temperature, json_mode,
            )
            choice = resp.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                raise TruncatedOutputError(
                    f"{provider.title()} hit its {max_tokens}-token output ceiling before finishing."
                )
            return choice.message.content

        elif provider == 'claude':
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)

            messages = [{"role": "user", "content": prompt}]
            # Prefilling the assistant turn with "{" is Anthropic's supported
            # way to force a bare JSON object — no fences, no preamble.
            if json_mode:
                messages.append({"role": "assistant", "content": "{"})

            create_kwargs = {
                "model": resolved_model,
                "max_tokens": max_tokens,
                "system": system_instruction,
                "messages": messages,
            }
            if temperature is not None:
                create_kwargs["temperature"] = temperature

            resp = client.messages.create(**create_kwargs)

            if getattr(resp, "stop_reason", None) == "max_tokens":
                raise TruncatedOutputError(
                    f"Claude hit its {max_tokens}-token output ceiling before finishing."
                )

            text = resp.content[0].text
            # The prefilled "{" is not echoed back — put it back on.
            return ("{" + text) if json_mode else text

        else:
            raise ValueError(f"Unsupported Provider: {provider}")

    except (ModelUnavailableError, TruncatedOutputError,
            RateLimitedError, QuotaExhaustedError):
        raise
    except Exception as e:
        err_str = str(e)
        status_hint = next(
            (str(getattr(e, a)) for a in ("status_code", "code", "http_status") if getattr(e, a, None) is not None),
            "no-status"
        )
        print(f"[call_llm] {provider}/{resolved_model} raised {type(e).__name__} "
              f"(status={status_hint}): {err_str[:200]}")

        # Order matters: check rate limits BEFORE model-availability, because
        # _is_model_unavailable() casts a deliberately wide net.
        kind = _classify_rate_limit(e)
        if kind == "daily":
            raise QuotaExhaustedError(
                f"{provider.title()} has cut you off for the rest of the day on model "
                f"'{resolved_model}' (free-tier daily request cap).\n\n"
                f"👉 Nothing generated so far is lost — this task is checkpointed. "
                f"Open it from the Story Library and press Resume once the quota "
                f"resets (midnight US Pacific for Gemini), or paste a different "
                f"provider's key in Engine Config and resume on that instead.\n\n"
                f"Vendor said: {err_str.splitlines()[0][:220]}"
            ) from e
        if kind == "transient":
            raise RateLimitedError(err_str[:300], retry_after=_extract_retry_after(err_str)) from e

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


# ==========================================
# 4b. PACING, BUDGET & RETRY WRAPPER
# ==========================================
# Every generation now goes through call_llm_guarded() instead of call_llm().
# It is the single place that owns "how often may we talk to a vendor, and
# what do we do when they say no".
_LAST_CALL_AT: dict[str, float] = {}
_CALL_LOCK = threading.Lock()


def _key_fingerprint(api_key: str) -> str:
    """Never log or key a dict on a raw API key."""
    return hashlib.sha256((api_key or "").encode("utf-8")).hexdigest()[:16]


def _throttle(api_key: str):
    """Guarantees MIN_SECONDS_BETWEEN_CALLS between two calls on the same key.
    Free-tier Gemini allows roughly 10 requests/minute; bursting the 8 chapter
    calls back-to-back is what produced the 'Please retry in 22s' flavour of
    429 even on days the daily quota was fine."""
    fp = _key_fingerprint(api_key)
    with _CALL_LOCK:
        wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - _LAST_CALL_AT.get(fp, 0.0))
        if wait > 0:
            time.sleep(wait)
        _LAST_CALL_AT[fp] = time.time()


class CallBudget:
    """A hard ceiling on how many requests ONE generation may spend.

    The point is that the creator finds out from Vystoria, with a clean
    message and an intact checkpoint, rather than from the vendor with a wall
    of protobuf. Set it a little under your daily allowance (e.g. 14 against
    Gemini's free 20) and a runaway retry loop can never drain the day."""

    def __init__(self, limit=0):
        self.limit = int(limit or 0)
        self.used = 0
        self.by_label: dict[str, int] = {}

    def charge(self, label: str):
        if self.limit and self.used >= self.limit:
            raise QuotaExhaustedError(
                f"This run hit its own budget of {self.limit} model calls "
                f"(spent on: {self.breakdown()}).\n\n"
                f"👉 Progress is checkpointed. Either raise 'Max model calls' in "
                f"Engine Config and press Resume, or resume tomorrow."
            )
        self.used += 1
        self.by_label[label] = self.by_label.get(label, 0) + 1

    def breakdown(self) -> str:
        return ", ".join(f"{k}×{v}" for k, v in sorted(self.by_label.items())) or "nothing"

    def summary(self) -> str:
        return f"{self.used} model call(s)" + (f" of {self.limit} budgeted" if self.limit else "")


def estimate_call_count(num_chapters: int, judge_mode: str) -> int:
    """What a clean run costs: world bible + outline + N chapters + manifest,
    plus the judge. Surfaced in the UI before the creator presses go."""
    calls = 2 + int(num_chapters) + 1
    if judge_mode != 'off':
        calls += 1
    if judge_mode == 'strict':
        calls += int(num_chapters) + 1   # worst case: one full regeneration
    return calls


def call_llm_guarded(prompt, system_instruction, provider, api_key, model_name,
                     *, label="call", budget=None, on_log=None, **llm_kwargs):
    """call_llm + pacing + budget + bounded back-off on transient 429s.

    Deliberately does NOT retry QuotaExhaustedError or ModelUnavailableError —
    both are terminal for this run, and retrying them is exactly what used to
    drain the daily allowance."""
    attempt = 0
    while True:
        attempt += 1
        if budget is not None:
            budget.charge(label)
        _throttle(api_key)

        try:
            return call_llm(prompt, system_instruction, provider, api_key,
                            model_name, **llm_kwargs)

        except RateLimitedError as rl:
            if attempt > MAX_RATE_LIMIT_RETRIES:
                raise QuotaExhaustedError(
                    f"{provider.title()} kept rate-limiting '{label}' through "
                    f"{MAX_RATE_LIMIT_RETRIES} back-offs. This is a per-minute cap, not a "
                    f"daily one, so it should clear shortly.\n\n"
                    f"👉 Progress is checkpointed — open the task from the Story Library "
                    f"and press Resume in a few minutes."
                ) from rl

            delay = rl.retry_after or (RATE_LIMIT_BASE_SLEEP * (2 ** (attempt - 1)))
            delay = min(delay, MAX_RATE_LIMIT_SLEEP) + random.uniform(0, 2)
            if on_log:
                on_log(f"⏳ {provider.title()} rate-limited '{label}'. Waiting {delay:.0f}s "
                       f"(back-off {attempt}/{MAX_RATE_LIMIT_RETRIES}) — no work lost.")
            time.sleep(delay)    

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


def validate_and_repair_scene_graph(all_scenes: list[dict]) -> list[str]:
    """Walks every next_scene_default and choices[].next_scene reference and
    makes sure it points at a scene that actually exists in this story. A
    dangling reference (typo'd id, hallucinated continuation, etc.) is
    treated as an ending — the field is stripped rather than left pointing
    at nothing. This is what stops a story from looping back to scene 1 or
    freezing when it hits a broken link near the end."""
    valid_ids = {s["id"] for s in all_scenes if s.get("id")}
    warnings = []

    for scene in all_scenes:
        sid = scene.get("id", "?")

        default_target = scene.get("next_scene_default")
        if default_target and default_target not in valid_ids:
            warnings.append(
                f"Scene '{sid}': next_scene_default '{default_target}' doesn't exist — "
                f"treating '{sid}' as an ending instead."
            )
            scene.pop("next_scene_default", None)

        if scene.get("choices"):
            kept = []
            for choice in scene["choices"]:
                target = choice.get("next_scene")
                if target and target in valid_ids:
                    kept.append(choice)
                else:
                    warnings.append(
                        f"Scene '{sid}': choice '{choice.get('text', '?')}' targets missing "
                        f"scene '{target}' — removing that choice."
                    )
            if kept:
                scene["choices"] = kept
            else:
                scene.pop("choices", None)
                scene.pop("choice_prompt", None)

    return warnings


def write_all_chapters(req, world_bible, outline, roster_prompt, canonical_names,
                       num_chapters, update_task, attempt_no, *, budget=None,
                       chapters_done=None, on_chapter_done=None,
                       scenes_per_chapter=14):
    """Runs the chapter-by-chapter generation loop once, start to finish.

    `chapters_done` is a {"1": [scene, ...], "2": [...]} map restored from the
    task's checkpoint. Any chapter present there is reused verbatim and costs
    ZERO model calls — this is what makes Resume cheap after a quota wall.

    `on_chapter_done(i, scenes)` is called after each freshly-written chapter
    so the caller can persist the checkpoint immediately. Losing eight
    chapters' worth of quota to a 429 on chapter nine was the old behaviour.

    Returns (all_scenes, starting_scene)."""
    chapters_done = dict(chapters_done or {})
    all_scenes = []
    starting_scene = None
    prev_last_scene = None
    previous_summary = "This is the very beginning."


    for i in range(1, num_chapters + 1):
        base_prog = 25 + int((i / num_chapters) * 55)
        key = str(i)

        cached = chapters_done.get(key)
        if cached:
            scenes = cached
            update_task('generating', f"Chapter {i} of {num_chapters} (restored)", base_prog,
                        f"♻️ Chapter {i} restored from checkpoint — 0 model calls spent.")
        else:
            step_msg = f"Writing Chapter {i} of {num_chapters} (attempt {attempt_no})..."
            update_task('generating', step_msg, base_prog, step_msg)

            if i == num_chapters:
                chapter_context = (
                    f"**IMPORTANT — this is the FINAL chapter ({i} of {num_chapters}).** "
                    "Any scene that represents a true story ending — including the very last "
                    "scene(s) in your 'scenes' array, and any earlier scene an early/bad choice "
                    "leads to that should terminate the story — MUST omit BOTH 'choices' AND "
                    "'next_scene_default' entirely. Do not invent a 'next_scene_default' id for an "
                    "ending scene; a scene with neither field is exactly how the game engine "
                    "recognizes 'The End'. Aim for the 3-5 distinct endings planned in the Outline, "
                    "with at least one reachable via the main path."
                )
            else:
                chapter_context = (
                    f"This is chapter {i} of {num_chapters}. The link from this chapter's final "
                    f"scene to chapter {i + 1}'s opening scene is wired up automatically after you "
                    "submit — just end the final scene linearly with any 'next_scene_default' "
                    "placeholder id; it will be overwritten, so don't worry about it being 'wrong'."
                )

            target_scenes = int(scenes_per_chapter)
            scenes = []
            last_error = None

            for attempt in range(1, CHAPTER_MAX_RETRIES + 1):
                prompt = CHAPTER_PROMPT.format(
                    chapter_number=i, chapter_context=chapter_context,
                    world_bible=world_bible, outline=outline,
                    previous_summary=previous_summary, roster=roster_prompt,
                    scene_count=target_scenes,
                )
                try:
                    raw_data = call_llm_guarded(
                        prompt,
                        "Output ONLY valid JSON. You MUST escape inner quotes like \\\"this\\\".",
                        req.provider, req.api_key, req.model_name,
                        label=f"chapter{i}", budget=budget,
                        on_log=lambda m: update_task('generating', step_msg, base_prog, m),
                        json_mode=True,
                    )
                    chapter_data = json.loads(clean_json_output(raw_data))
                    scenes = [s for s in chapter_data.get("scenes", []) if s.get("id")]
                    if scenes:
                        update_task('generating', step_msg, base_prog,
                                    f"Chapter {i} structured and validated ({len(scenes)} scenes).")
                        break
                    last_error = "the model returned zero usable scenes"

                except (ModelUnavailableError, QuotaExhaustedError):
                    # Terminal — never retried. Re-raised so the pipeline can
                    # checkpoint and stop cleanly instead of burning the day.
                    raise

                except TruncatedOutputError as te:
                    last_error = str(te)
                    target_scenes = max(6, int(target_scenes * 0.6))
                    update_task('generating', step_msg, base_prog,
                                f"⚠️ Chapter {i} overran the output-token ceiling. "
                                f"Retrying with {target_scenes} scenes instead of the full ask.")
                    continue

                except json.JSONDecodeError as je:
                    last_error = f"invalid JSON ({je})"
                    update_task('generating', step_msg, base_prog,
                                f"⚠️ Chapter {i} attempt {attempt}/{CHAPTER_MAX_RETRIES} "
                                f"returned malformed JSON. Retrying...")

                except Exception as e:
                    last_error = str(e)[:200]
                    update_task('generating', step_msg, base_prog,
                                f"⚠️ Chapter {i} attempt {attempt}/{CHAPTER_MAX_RETRIES} failed "
                                f"({last_error}). Retrying...")

                time.sleep(2)

            if not scenes:
                raise Exception(
                    f"Chapter {i} could not be generated after {CHAPTER_MAX_RETRIES} attempts. "
                    f"Last error: {last_error}"
                )

            # Drop duplicate ids inside the chapter before they can poison the
            # scene graph (a repeated id makes the player's find() ambiguous).
            seen_ids, deduped = set(), []
            for s in scenes:
                if s["id"] in seen_ids:
                    continue
                seen_ids.add(s["id"])
                deduped.append(s)
            scenes = deduped

            scenes = normalize_speakers(scenes, canonical_names)
            chapters_done[key] = scenes
            if on_chapter_done:
                on_chapter_done(i, scenes)

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

    repair_warnings = validate_and_repair_scene_graph(all_scenes)
    for w in repair_warnings:
        update_task('generating', f"Validating story structure (attempt {attempt_no})...", 82, f"⚠️ {w}")

    return all_scenes, (starting_scene or "ch1_scene01")

def build_asset_manifest(world_bible, speaker_expressions_map, background_ids,
                         provider, api_key, model_name, *, budget=None, on_log=None):
    """Catalogs character/background/cover art descriptions with per-expression variants.

    Never raises for ordinary failures — falls back to a bare-name manifest —
    but a quota wall IS re-raised, because silently degrading to an empty
    manifest hides the real reason from the creator."""
    speaker_expressions_text = "\n".join(
        f"- {name}: {', '.join(sorted(exprs)) or 'neutral'}"
        for name, exprs in sorted(speaker_expressions_map.items())
    ) or "(none)"

    try:
        manifest_raw = call_llm_guarded(
            ASSET_MANIFEST_PROMPT.format(
                world_bible=world_bible,
                speaker_expressions=speaker_expressions_text,
                backgrounds="\n".join(f"- {b}" for b in background_ids) or "(none)"
            ),
            "Output ONLY valid JSON.", provider, api_key, model_name,
            label="asset-manifest", budget=budget, on_log=on_log, json_mode=True,
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
        # Both ride along in the manifest call — no extra model request.
        # `synopsis` becomes stories.description (what the player app shows on
        # the featured card and the detail screen); `art_direction` pre-fills
        # the House Art Style box so every asset shares one look.
        asset_manifest.setdefault("synopsis", "")
        asset_manifest.setdefault("art_direction", "")
        return asset_manifest, None

    except (QuotaExhaustedError, ModelUnavailableError):
        raise
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
            "cover": {"description": ""},
            "synopsis": "",
            "art_direction": "",
        }
        return fallback, str(e)


def _reachable_scene_ids(final_story):
    """BFS from starting_scene. Anything outside the result was written but can
    never be played — a real defect the judge has no way to notice by reading."""
    by_id = {s["id"]: s for s in final_story.get("scenes", []) if s.get("id")}
    start = final_story.get("starting_scene") or next(iter(by_id), None)
    seen, stack = set(), ([start] if start else [])
    while stack:
        sid = stack.pop()
        if not sid or sid in seen or sid not in by_id:
            continue
        seen.add(sid)
        scene = by_id[sid]
        if scene.get("next_scene_default"):
            stack.append(scene["next_scene_default"])
        for choice in scene.get("choices") or []:
            stack.append(choice.get("next_scene"))
    return seen


def compute_story_stats(final_story):
    """Measures everything measurable in Python so the judge doesn't have to
    count — and can't get the counting wrong. Cheap, deterministic, free."""
    scenes = final_story.get("scenes", [])
    all_ids = {s.get("id") for s in scenes if s.get("id")}
    reachable = _reachable_scene_ids(final_story)

    dialogue = narrative = 0
    choice_scenes = total_choices = 0
    generic_prompts = []
    long_choices = []
    speakers, expressions, endings = {}, {}, []

    for scene in scenes:
        for block in scene.get("sequence", []):
            if block.get("type") == "dialogue":
                dialogue += 1
                spk = block.get("speaker") or "(unnamed)"
                speakers[spk] = speakers.get(spk, 0) + 1
                exp = block.get("expression") or "neutral"
                expressions[exp] = expressions.get(exp, 0) + 1
            else:
                narrative += 1

        if scene.get("choices"):
            choice_scenes += 1
            total_choices += len(scene["choices"])
            prompt = (scene.get("choice_prompt") or "").strip()
            if len(prompt) < 20:
                generic_prompts.append(scene.get("id"))
            for choice in scene["choices"]:
                if len((choice.get("text") or "").split()) > 7:
                    long_choices.append(f'{scene.get("id")}:"{choice.get("text")}"')
        elif not scene.get("next_scene_default"):
            endings.append(scene.get("id"))

    total_blocks = (dialogue + narrative) or 1
    stats = {
        "scene_count": len(scenes),
        "dialogue_blocks": dialogue,
        "narrative_blocks": narrative,
        "dialogue_ratio": round(dialogue / total_blocks, 3),
        "choice_scenes": choice_scenes,
        "total_choices": total_choices,
        "scenes_per_choice": round(len(scenes) / choice_scenes, 2) if choice_scenes else None,
        "ending_count": len(endings),
        "endings": endings[:12],
        "unreachable_scenes": sorted(all_ids - reachable)[:15],
        "unreachable_count": len(all_ids - reachable),
        "speaker_line_counts": dict(sorted(speakers.items(), key=lambda kv: -kv[1])[:15]),
        "expression_usage": dict(sorted(expressions.items(), key=lambda kv: -kv[1])),
        "scenes_with_thin_choice_prompt": generic_prompts[:10],
        "choices_over_7_words": long_choices[:10],
    }
    return stats


def build_judge_digest(final_story, max_full_scenes=28, max_chars=45000):
    """Compact, judge-readable rendering of the whole story.

    The old judge received json.dumps() of every scene — on an 8-chapter book
    that is well past what a Flash-tier model reads reliably, which is why the
    scorecard so often came back as ERROR or with hallucinated scene ids. Full
    text for an evenly-spaced sample; one summary line for the rest; every
    link and choice for all of them."""
    scenes = final_story.get("scenes", [])
    if not scenes:
        return "(story is empty)"

    step = max(1, len(scenes) // max_full_scenes)
    sampled = set(list(range(0, len(scenes), step))[:max_full_scenes])

    lines = []
    for idx, scene in enumerate(scenes):
        header = f"### {scene.get('id')}  [bg: {scene.get('background', '—')}]"
        sequence = scene.get("sequence", [])

        if idx in sampled:
            lines.append(header + "   (full text)")
            for block in sequence:
                if block.get("type") == "dialogue":
                    lines.append(f'   {block.get("speaker", "?")} '
                                 f'({block.get("expression", "neutral")}): {block.get("text", "")}')
                else:
                    lines.append(f'   [narration] {block.get("text", "")}')
        else:
            d = sum(1 for b in sequence if b.get("type") == "dialogue")
            n = len(sequence) - d
            spk = ", ".join(sorted({b.get("speaker") for b in sequence if b.get("speaker")}))
            lines.append(f"{header}   {d} dialogue / {n} narrative · speakers: {spk or '—'}")

        if scene.get("choices"):
            lines.append(f'   CHOICE PROMPT: {scene.get("choice_prompt") or "(MISSING)"}')
            for choice in scene["choices"]:
                lines.append(f'      -> "{choice.get("text")}"  ==> {choice.get("next_scene")}')
        elif scene.get("next_scene_default"):
            lines.append(f'   -> {scene["next_scene_default"]}')
        else:
            lines.append("   -> [ENDING]")

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(digest truncated)"
    return text

def run_judge_evaluation(world_bible, final_story, provider, api_key, model_name,
                         *, budget=None, on_log=None):
    """Runs the LLM-as-a-Judge QA stage and returns (scorecard, error).

    Never raises for ordinary failures — a judge that can't parse its own JSON
    must not sink a story that generated fine. Quota walls DO propagate, so
    the pipeline can checkpoint rather than mislabel the run."""
    try:
        raw = call_llm_guarded(
            JUDGE_PROMPT.format(
                world_bible=world_bible,
                story_stats=json.dumps(compute_story_stats(final_story), ensure_ascii=False, indent=2),
                story_digest=build_judge_digest(final_story),
            ),
            "You are a rigorous, detail-oriented Visual Novel quality judge. Output ONLY valid JSON.",
            provider, api_key, model_name,
            label="judge", budget=budget, on_log=on_log, json_mode=True,
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
        status = "FAIL" if (overall_score is None or overall_score < JUDGE_PASS_SCORE or failed_params) else "PASS"

        return {
            "overall_score": overall_score,
            "status": status,
            "failed_parameters": failed_params,
            "summary": evaluation.get("summary", ""),
            "metrics": metrics,
            "actionable_critiques": evaluation.get("actionable_critiques", []),
            "structural_stats": compute_story_stats(final_story),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }, None

    except (QuotaExhaustedError, ModelUnavailableError):
        raise
    except Exception as e:
        return {
            "overall_score": None,
            "status": "ERROR",
            "failed_parameters": [],
            "summary": f"AI evaluation could not be completed automatically: {e}",
            "metrics": {},
            "actionable_critiques": [],
            "structural_stats": compute_story_stats(final_story),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }, str(e)

def tweak_scene(world_bible, scene, instruction, provider, api_key, model_name):
    """Rewrites ONE scene per a targeted creator instruction."""
    original_id = scene.get("id")
    raw = call_llm_guarded(
        TWEAK_PROMPT.format(
            world_bible=world_bible or "(none provided)",
            scene_json=json.dumps(scene, ensure_ascii=False),
            instruction=instruction
        ),
        "Output ONLY valid JSON for the single revised scene. Never change 'id' or invent new scene ids.",
        provider, api_key, model_name,
        label="tweak-scene", json_mode=True,
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

# ==========================================
# 5a. CHECKPOINTING
# ==========================================
# The whole point: a quota wall on chapter 7 must not cost the six chapters
# already paid for. Every expensive artifact lands in generation_tasks.checkpoint
# the moment it exists, and Resume replays from there for free.
def _load_checkpoint(task_id: str) -> dict:
    try:
        res = supabase.table("generation_tasks").select("checkpoint").eq("id", task_id).single().execute()
        checkpoint = (res.data or {}).get("checkpoint") or {}
        if checkpoint.get("version") != CHECKPOINT_VERSION:
            return {}
        return checkpoint
    except Exception:
        traceback.print_exc()
        return {}


def _save_checkpoint(task_id: str, checkpoint: dict):
    """Writes the full checkpoint plus a SMALL summary column. The frontend
    polls every 2s and must not drag a megabyte of scene JSON down each time,
    so it reads checkpoint_progress and never `checkpoint` itself."""
    checkpoint["version"] = CHECKPOINT_VERSION
    chapters = checkpoint.get("chapters") or {}
    summary = {
        "has_world_bible": bool(checkpoint.get("world_bible")),
        "has_outline": bool(checkpoint.get("outline")),
        "chapters_done": len(chapters),
        "num_chapters": checkpoint.get("num_chapters"),
        "scenes_banked": sum(len(v or []) for v in chapters.values()),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        supabase.table("generation_tasks").update({
            "checkpoint": checkpoint,
            "checkpoint_progress": summary,
        }).eq("id", task_id).execute()
    except Exception as db_err:
        print(f"[{task_id}] ⚠️ Failed to save checkpoint: {db_err}")


def _set_failure_kind(task_id: str, kind: str | None):
    """'quota' unlocks the Resume button in the creator app; 'model' points the
    creator at Engine Config; None clears it on a fresh start."""
    try:
        supabase.table("generation_tasks").update({"failure_kind": kind}).eq("id", task_id).execute()
    except Exception as db_err:
        print(f"[{task_id}] ⚠️ Failed to set failure_kind: {db_err}")


def run_generation_pipeline(task_id: str, req, resume: bool = False):
    """Writes one complete visual novel, checkpointing as it goes.

    Call-count arithmetic for an 8-chapter book, which is what made the free
    tier unusable before:

        old worst case   1 world + 1 outline + (8 × 3 retries) + 1 judge
                         + 8 regenerated chapters + 1 judge + 1 manifest  ≈ 30+
        new typical      1 world + 1 outline + 8 chapters + 1 judge
                         + 1 manifest                                     = 12
        new, resumed     only the chapters not already in the checkpoint.
    """
    logs = []

    def update_task(status, current_step, progress, log_msg=None, final_url=None):
        if log_msg:
            print(f"[{task_id}] {log_msg}")
            logs.append(log_msg)

        payload = {
            "status": status,
            "current_step": current_step,
            "progress_percent": progress,
            "logs": logs[-200:],
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        if final_url:
            payload["final_url"] = final_url

        try:
            supabase.table("generation_tasks").update(payload).eq("id", task_id).execute()
        except Exception as db_err:
            print(f"[{task_id}] ⚠️ Failed to write task update to Supabase: {db_err}")
            traceback.print_exc()

    

    checkpoint = _load_checkpoint(task_id) if resume else {}
    budget = CallBudget(getattr(req, "max_llm_calls", 0))
    judge_mode = (getattr(req, "judge_mode", "advisory") or "advisory").lower()
    scenes_per_chapter = int(getattr(req, "scenes_per_chapter", 14) or 14)

    _set_failure_kind(task_id, None)

    try:
        num_chapters = checkpoint.get("num_chapters")
        if not num_chapters:
            num_chapters = 8
            match = re.search(r'\d+', getattr(req, "target_length", "") or "")
            if match:
                num_chapters = int(match.group())

        chapters_done = checkpoint.get("chapters") or {}
        estimated = estimate_call_count(num_chapters, judge_mode) - len(chapters_done)

        if resume:
            update_task('generating', 'Resuming from checkpoint...', 5,
                        f"♻️ Resuming task. {len(chapters_done)}/{num_chapters} chapters already banked — "
                        f"about {max(1, estimated)} model call(s) left to pay for.")
        else:
            update_task('generating', 'Building World Bible...', 5,
                        f"Started {req.provider.upper()} engine using model '{req.model_name or '(backend default)'}'. "
                        f"Chapters: {num_chapters} × {scenes_per_chapter} scenes. Judge: {judge_mode}. "
                        f"Estimated cost: ~{estimated} model calls"
                        + (f" (budget {budget.limit})." if budget.limit else "."))

        # ---------- WORLD BIBLE ----------
        world_bible = checkpoint.get("world_bible")
        naming_pool = checkpoint.get("naming_pool") or pick_naming_pool(3)

        if world_bible:
            update_task('generating', 'World Bible restored.', 12,
                        "♻️ World Bible restored from checkpoint — 0 model calls spent.")
        else:
            idea_section = (
                f"- Core Idea (build the plot and world firmly around this creator-provided concept): {req.idea}"
                if getattr(req, "idea", None) and req.idea.strip() else ""
            )
            has_reference = bool(getattr(req, "reference_text", None) and req.reference_text.strip())
            trimmed_reference = (getattr(req, "reference_text", "") or "").strip()[:MAX_REFERENCE_CHARS]
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
            if has_reference:
                update_task('generating', 'Building World Bible...', 5,
                            f"Reference document detected ({len(trimmed_reference)} chars) — adapting it "
                            f"instead of freeform generation.")

            world_bible = call_llm_guarded(
                WORLD_PROMPT.format(title=req.title, subtitle=req.subtitle, genre=req.genre,
                                    target_length=req.target_length, tone=req.tone,
                                    idea_section=idea_section, reference_section=reference_section,
                                    naming_pool=naming_pool),
                "You are a master visual novel author.",
                req.provider, req.api_key, req.model_name,
                label="world-bible", budget=budget,
                on_log=lambda m: update_task('generating', 'Building World Bible...', 5, m),
            )
            checkpoint.update({
                "world_bible": world_bible,
                "naming_pool": naming_pool,
                "num_chapters": num_chapters,
                "title": req.title,
                "subtitle": req.subtitle,
                "genre": req.genre,
                "target_length": req.target_length,
                "tone": req.tone,
                "idea": getattr(req, "idea", None),
                "reference_text": (getattr(req, "reference_text", "") or "")[:MAX_REFERENCE_CHARS],
            })
            _save_checkpoint(task_id, checkpoint)

        roster_prompt, canonical_names, expressions_by_char = parse_character_roster(world_bible)
        if canonical_names:
            update_task('generating', 'World Bible ready.', 15,
                        f"Roster locked in with {len(canonical_names)} canonical character(s): "
                        f"{', '.join(sorted(canonical_names))}.")
        else:
            update_task('generating', 'World Bible ready.', 15,
                        "⚠️ No roster block found in World Bible — speaker names won't be normalized.")

        # ---------- OUTLINE ----------
        outline = checkpoint.get("outline")
        if outline:
            update_task('generating', 'Outline restored.', 22,
                        "♻️ Outline restored from checkpoint — 0 model calls spent.")
        else:
            idea_reminder = (
                f"- Stay faithful to this core idea from the creator: {req.idea}"
                if getattr(req, "idea", None) and req.idea.strip() else ""
            )
            reference_reminder = (
                "- This outline MUST follow the plot/structure of the Reference Document supplied when "
                "building the World Bible — do not diverge from it."
                if checkpoint.get("reference_text") else ""
            )
            outline = call_llm_guarded(
                OUTLINE_PROMPT.format(target_length=req.target_length, world_bible=world_bible,
                                      idea_reminder=idea_reminder, reference_reminder=reference_reminder),
                "You are a master visual novel author.",
                req.provider, req.api_key, req.model_name,
                label="outline", budget=budget,
                on_log=lambda m: update_task('generating', 'Building Outline...', 20, m),
            )
            checkpoint["outline"] = outline
            _save_checkpoint(task_id, checkpoint)

        # ---------- CHAPTERS (+ optional strict re-roll) ----------
        max_judge_attempts = 2 if judge_mode == "strict" else 1

        update_task('generating', 'Writing Chapters...', 25,
                    f"Master outline locked in. Beginning chapter pipeline "
                    f"({len(checkpoint.get('chapters') or {})}/{num_chapters} already banked).")

        final_story = None
        evaluation_scorecard = None
        judge_err = None
        all_scenes, starting_scene = [], None

        def on_chapter_done(index, scenes):
            checkpoint.setdefault("chapters", {})[str(index)] = scenes
            _save_checkpoint(task_id, checkpoint)

        for judge_attempt in range(1, max_judge_attempts + 1):
            all_scenes, starting_scene = write_all_chapters(
                req, world_bible, outline, roster_prompt, canonical_names,
                num_chapters, update_task, judge_attempt,
                budget=budget,
                chapters_done=checkpoint.get("chapters") or {},
                on_chapter_done=on_chapter_done,
                scenes_per_chapter=scenes_per_chapter,
            )

            final_story = {
                "title": f"{req.title}: {req.subtitle}",
                "starting_scene": starting_scene,
                "scenes": all_scenes
            }
            checkpoint["final_story"] = final_story
            _save_checkpoint(task_id, checkpoint)

            if judge_mode == "off":
                update_task('generating', 'Judge skipped', 86,
                            "⏭️ AI Judge is switched off for this run (Engine Config). "
                            "Saved one model call.")
                break

            update_task('generating', f'Running AI Judge (pass {judge_attempt}/{max_judge_attempts})...', 85,
                        "Submitting a structural digest of the draft to the AI Judge...")
            evaluation_scorecard, judge_err = run_judge_evaluation(
                world_bible, final_story, req.provider, req.api_key, req.model_name,
                budget=budget, on_log=lambda m: update_task('generating', 'Running AI Judge...', 85, m),
            )

            if judge_err:
                update_task('generating', 'AI evaluation could not complete.', 87,
                            f"⚠️ AI Judge could not run ({judge_err}). The story itself is fine — "
                            f"re-run the judge from the AI Judgement screen whenever you like.")
                break

            score = evaluation_scorecard.get('overall_score')
            if evaluation_scorecard['status'] == 'PASS':
                update_task('generating', 'AI Judge: PASS', 87,
                            f"🧑‍⚖️ Judge verdict: PASS"
                            + (f" (Weighted Score: {score}/10)" if score is not None else "") + ".")
                break

            failed = ', '.join(evaluation_scorecard.get('failed_parameters', [])) or 'n/a'
            if judge_mode != "strict":
                update_task('generating', 'AI Judge: FAIL (advisory)', 87,
                            f"🧑‍⚖️ Judge verdict: FAIL"
                            + (f" (Score: {score}/10)" if score is not None else "")
                            + f". Weak parameters: {failed}. Advisory mode — the draft is kept as-is. "
                              f"Use Tweak Scene on the specific scenes named in the critiques instead "
                              f"of paying for a whole regeneration.")
                break

            if judge_attempt < max_judge_attempts:
                # Strict mode only: throw the chapters away and re-roll.
                checkpoint["chapters"] = {}
                _save_checkpoint(task_id, checkpoint)
                update_task('generating', 'AI Judge: FAIL — regenerating', 87,
                            f"🧑‍⚖️ Judge verdict: FAIL"
                            + (f" (Score: {score}/10)" if score is not None else "")
                            + f". Weak parameters: {failed}. Strict mode — regenerating all chapters "
                              f"(this costs another {num_chapters} model calls).")
            else:
                update_task('generating', 'AI Judge: FAIL (max attempts reached)', 87,
                            f"🧑‍⚖️ Judge verdict: FAIL after {max_judge_attempts} attempts"
                            + (f" (Score: {score}/10)" if score is not None else "")
                            + ". Proceeding with the last draft — review the scorecard and use Tweak Scene.")

        # ---------- ASSET MANIFEST ----------
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

        update_task('generating', 'Cataloging assets...', 90,
                    f"Found {len(speaker_expressions_map)} unique speakers across "
                    f"{sum(len(v) for v in speaker_expressions_map.values())} portrait variants.")

        asset_manifest = checkpoint.get("asset_manifest")
        asset_err = None
        if asset_manifest:
            update_task('generating', 'Asset manifest restored.', 92,
                        "♻️ Asset manifest restored from checkpoint — 0 model calls spent.")
        else:
            asset_manifest, asset_err = build_asset_manifest(
                world_bible, speaker_expressions_map, background_ids,
                req.provider, req.api_key, req.model_name,
                budget=budget,
                on_log=lambda m: update_task('generating', 'Cataloging assets...', 90, m),
            )
            checkpoint["asset_manifest"] = asset_manifest
            _save_checkpoint(task_id, checkpoint)

        if asset_err:
            update_task('generating', 'Asset manifest ready (fallback).', 92,
                        f"⚠️ Could not auto-describe assets ({asset_err}); showing names/expressions only.")
        else:
            total_variants = sum(len(c.get('expressions', [])) for c in asset_manifest.get('characters', []))
            update_task('generating', 'Asset manifest ready.', 92,
                        f"Cataloged {len(asset_manifest.get('characters', []))} characters "
                        f"({total_variants} portrait variants) and {len(background_ids)} backgrounds.")

        # ---------- PERSIST ----------
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
                        f"⚠️ Story saved fine, but couldn't save the AI Judge scorecard: {db_err}")

        update_task('completed', 'Ready for review', 100,
                    f"✅ Story generated in {budget.summary()} ({budget.breakdown()}). "
                    f"Play-test it end-to-end to unlock publishing.")

    except QuotaExhaustedError as qee:
        # The headline fix. Everything paid for so far is already in the
        # checkpoint, so this is a pause, not a loss.
        traceback.print_exc()
        _save_checkpoint(task_id, checkpoint)
        _set_failure_kind(task_id, "quota")
        banked = len((checkpoint.get("chapters") or {}))
        update_task('failed', 'Paused — provider quota reached', 0,
                    f"⏸️ {str(qee)}\n\nCheckpoint holds "
                    f"{'a World Bible, ' if checkpoint.get('world_bible') else ''}"
                    f"{'an Outline, ' if checkpoint.get('outline') else ''}"
                    f"{banked} finished chapter(s). Spent this run: {budget.summary()}.")

    except ModelUnavailableError as mue:
        traceback.print_exc()
        _save_checkpoint(task_id, checkpoint)
        _set_failure_kind(task_id, "model")
        update_task('failed', 'Model unavailable', 0, f"❌ {str(mue)}")

    except Exception as e:
        traceback.print_exc()
        _save_checkpoint(task_id, checkpoint)
        _set_failure_kind(task_id, "other")
        update_task('failed', 'Error occurred', 0,
                    f"❌ FATAL ERROR: {str(e)}\n\nAnything already generated is checkpointed — "
                    f"press Resume rather than starting over.")

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
            "current_step": "Initializing...",
            "checkpoint": None,
            "checkpoint_progress": None,
            "failure_kind": None,
        }).execute()

        if not res.data:
            raise Exception("Insert returned no data — check that the 'generation_tasks' table exists and RLS policies allow this insert.")

        task_id = res.data[0]["id"]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to create generation task: {str(e)}")

    background_tasks.add_task(run_generation_pipeline, task_id, req, False)

    num_chapters = 8
    match = re.search(r'\d+', req.target_length or "")
    if match:
        num_chapters = int(match.group())

    return {
        "status": "success",
        "message": "Generation task started in the background.",
        "task_id": task_id,
        "estimated_calls": estimate_call_count(num_chapters, req.judge_mode),
    }


@app.post("/resume/{task_id}")
def resume_story_endpoint(task_id: str, req: ResumeRequest, background_tasks: BackgroundTasks):
    """Picks a checkpointed task back up. The provider/key/model may be
    different from the original run — which is the whole point when the
    original key is out of quota for the day."""
    try:
        res = supabase.table("generation_tasks").select(
            "id, title, checkpoint, status"
        ).eq("id", task_id).single().execute()
        row = res.data
        if not row:
            raise HTTPException(status_code=404, detail="No such generation task.")

        checkpoint = row.get("checkpoint") or {}
        if checkpoint.get("version") != CHECKPOINT_VERSION or not checkpoint.get("world_bible"):
            raise HTTPException(
                status_code=400,
                detail="This task has no usable checkpoint — nothing to resume from. "
                       "Start a fresh generation instead."
            )
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not load the task: {str(e)}")

    # Rebuild a request object out of the checkpoint so the original brief is
    # honoured even though the caller only sent credentials.
    resume_req = GenerateRequest(
        provider=req.provider,
        api_key=req.api_key,
        model_name=req.model_name,
        title=checkpoint.get("title") or (row.get("title") or "Untitled").split(":")[0].strip(),
        subtitle=checkpoint.get("subtitle") or "",
        genre=checkpoint.get("genre") or "",
        target_length=checkpoint.get("target_length") or f"{checkpoint.get('num_chapters', 8)} chapters",
        tone=checkpoint.get("tone") or "",
        idea=checkpoint.get("idea"),
        reference_text=checkpoint.get("reference_text"),
        user_id="",  # unused on resume — the row already exists
        judge_mode=req.judge_mode,
        scenes_per_chapter=req.scenes_per_chapter,
        max_llm_calls=req.max_llm_calls,
    )

    supabase.table("generation_tasks").update({
        "status": "pending",
        "current_step": "Resuming...",
        "failure_kind": None,
        "provider": req.provider,
    }).eq("id", task_id).execute()

    background_tasks.add_task(run_generation_pipeline, task_id, resume_req, True)

    done = len(checkpoint.get("chapters") or {})
    total = checkpoint.get("num_chapters") or 8
    return {
        "status": "success",
        "task_id": task_id,
        "chapters_restored": done,
        "chapters_remaining": max(0, total - done),
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
            world_bible, final_story, req.provider, req.api_key, req.model_name,
            budget=CallBudget(5),   # a manual re-run should never spiral
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


# ==========================================
# 6b. ASSET IMAGE GENERATION
# ==========================================
# Deliberately a separate provider + key from the text engine. Writing a story
# on Gemini's free tier and then spending the same 20-request allowance on 40
# character portraits is exactly how the day's quota disappears.
# Gemini's image models take a real aspect_ratio setting, so these are exact.
IMAGE_ASPECTS = {
    "character":  "3:4",    # tall — the engine draws portraits at h-[80%], bottom-anchored
    "background": "16:9",   # matches the play area
    "cover":      "4:3",    # see the safe-zone note in ASPECT_HINTS["cover"]
}

# OpenAI only offers three sizes, so these are the nearest fit. A cover comes
# back square rather than 4:3 — which is fine, because the safe-zone rule below
# already forces the important content into a centred square.
IMAGE_SIZES = {
    "openai": {
        "character":  "1024x1536",
        "background": "1536x1024",
        "cover":      "1024x1024",
    },
    "dalle": {
        "character":  "1024x1792",
        "background": "1792x1024",
        "cover":      "1024x1024",
    },
}

# Shared negative prompt. The single most common complaint about generated VN
# art is that it comes back looking like a blurry 3D render with a shallow
# depth of field — image models default to "cinematic" unless told otherwise.
# Naming the failure modes explicitly is what stops it.
FLAT_2D_RULES = (
    "Flat 2D illustration with clean crisp line art and hard-edged cel shading, "
    "fully in focus from edge to edge. "
    "NOT 3D, NOT a render, NOT photorealistic, NOT a photograph. "
    "No depth-of-field blur, no bokeh, no motion blur, no soft focus, no haze, "
    "no film grain, no noise, no lens flare, no chromatic aberration, no vignette. "
    "No text, no lettering, no title, no logo, no watermark, no signature, "
    "no border, no frame, no UI elements."
)

ASPECT_HINTS = {
    # Portraits are composited over a background at runtime, so anything behind
    # the figure is something the creator has to cut out by hand later.
    "character": (
        "Full-body character reference of ONE single figure, standing, facing the viewer, "
        "in a neutral relaxed pose. Vertical 3:4 framing, figure centred, with the whole "
        "body from the top of the head to the soles of the feet inside the frame and clear "
        "margin above and below — do not crop the head or the feet. "
        "COMPLETELY TRANSPARENT BACKGROUND. Nothing at all behind the figure: no scenery, "
        "no room, no floor, no ground, no cast shadow, no drop shadow, no colour fill, "
        "no gradient, no backdrop, no props. Clean sharp silhouette edges, ready to cut out "
        "and composite over a scene. "
        "One figure only — no turnaround sheet, no multiple poses, no side views, "
        "no reference grid, no colour swatches, no speech bubbles. "
        + FLAT_2D_RULES
    ),
    # At runtime a dialogue box covers the bottom of the screen and a character
    # portrait stands on the right, so detail placed there is never seen.
    "background": (
        "Empty environment artwork, 16:9 landscape, wide establishing shot at roughly "
        "eye level. "
        "ABSOLUTELY NO PEOPLE: no characters, no figures, no silhouettes, no crowds, "
        "no animals, no faces. This is an empty stage that characters are drawn on top of. "
        "Compose the important detail in the upper two thirds and the left half of the "
        "frame: at runtime the bottom third is covered by the dialogue box and the right "
        "third by a character portrait. Keep those regions visually quiet. "
        + FLAT_2D_RULES
    ),
    # The app crops this to 3:4, 1:1, 8:7 and 16:9 in four different places.
    # A centred square safe zone is the only composition that survives all four.
    "cover": (
        "Poster-style key art, 4:3, with ONE clear focal subject placed dead centre. "
        "CRITICAL SAFE ZONE: every essential element — the subject's face, the focal "
        "object, the silhouette — must sit inside a centred square occupying the middle "
        "of the frame. The app crops this image to tall 3:4, square 1:1 and wide 16:9 in "
        "different screens, so anything near the left or right edges or the extreme top "
        "or bottom WILL be cut off. Treat the outer margins as atmosphere only. "
        "Bold readable silhouette that still works shrunk to a thumbnail. "
        "Leave the image completely free of lettering — the app draws the title itself. "
        + FLAT_2D_RULES
    ),
}


def _compose_image_prompt(req: ImageRequest) -> str:
    """Subject first, then house style, then composition and negatives.

    If the caller already composed the prompt (the creator app does), return it
    verbatim — appending a second copy of the rules would both waste tokens and
    let the two wordings drift apart."""
    if not req.compose:
        return req.prompt.strip()

    parts = [req.prompt.strip()]
    if req.style and req.style.strip():
        parts.append(f"House art style for this entire project: {req.style.strip()}")
    parts.append(ASPECT_HINTS.get(req.kind, ASPECT_HINTS["character"]))
    return "\n\n".join(parts)


def _coerce_image_bytes(data):
    """Different SDK versions hand back raw bytes or an already-base64 str."""
    if isinstance(data, bytes):
        return base64.b64encode(data).decode("utf-8")
    if isinstance(data, str):
        return data
    raise Exception("Image payload was neither bytes nor base64 text.")


def _gemini_image(api_key, model, prompt, kind="character"):
    aspect = IMAGE_ASPECTS.get(kind, "1:1")

    try:
        from google import genai as google_genai
        from google.genai import types as google_types

        client = google_genai.Client(api_key=api_key)

        # image_config lands in different SDK versions at different times, so
        # try it and fall back to a plain call rather than hard-failing on an
        # older google-genai. Without it Gemini returns 1:1 and the framing
        # rules in the prompt are all the composition control we get.
        config_attempts = []
        try:
            config_attempts.append(google_types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=google_types.ImageConfig(aspect_ratio=aspect),
            ))
        except (AttributeError, TypeError):
            pass
        config_attempts.append(google_types.GenerateContentConfig(response_modalities=["IMAGE"]))

        last_error = None
        for config in config_attempts:
            try:
                resp = client.models.generate_content(model=model, contents=prompt, config=config)
                for part in resp.candidates[0].content.parts:
                    inline = getattr(part, "inline_data", None)
                    if inline and getattr(inline, "data", None):
                        return _coerce_image_bytes(inline.data), (getattr(inline, "mime_type", None) or "image/png")
                last_error = Exception("Gemini returned no image part.")
            except Exception as e:
                # A rate limit is terminal — don't burn a second request on the
                # fallback config just to hit the same wall.
                if _classify_rate_limit(e):
                    raise
                last_error = e
        raise last_error or Exception("Gemini returned no image part.")

    except ImportError:
        pass

    # Legacy google-generativeai fallback. No aspect-ratio control here.
    import google.generativeai as legacy
    legacy.configure(api_key=api_key)
    resp = legacy.GenerativeModel(model).generate_content(prompt)
    for candidate in resp.candidates or []:
        for part in candidate.content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and getattr(inline, "data", None):
                return _coerce_image_bytes(inline.data), (getattr(inline, "mime_type", None) or "image/png")
    raise Exception(
        "This Gemini model did not return an image. Use an image-capable model id "
        f"(e.g. {DEFAULT_IMAGE_MODELS['gemini']}), and `pip install -U google-genai`."
    )


def _openai_image(api_key, model, prompt, kind):
    import openai
    client = openai.OpenAI(api_key=api_key)

    family = "dalle" if "dall-e" in (model or "").lower() else "openai"
    size = IMAGE_SIZES[family].get(kind, "1024x1024")

    kwargs = {"model": model, "prompt": prompt, "size": size, "n": 1}

    if family == "dalle":
        kwargs["response_format"] = "b64_json"
    elif kind == "character":
        # gpt-image-1 can return a genuinely transparent alpha channel, which
        # is far better than asking for a "plain background" and then cutting a
        # flat colour out by hand. Requires png (or webp) output.
        kwargs["background"] = "transparent"
        kwargs["output_format"] = "png"

    try:
        resp = client.images.generate(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        # Older accounts/models reject `background` or `output_format`. Drop the
        # optional extras and retry once rather than failing the whole request.
        if "background" in msg or "output_format" in msg:
            kwargs.pop("background", None)
            kwargs.pop("output_format", None)
            resp = client.images.generate(**kwargs)
        elif "size" in msg:
            kwargs["size"] = "1024x1024"
            resp = client.images.generate(**kwargs)
        else:
            raise

    item = resp.data[0]
    if getattr(item, "b64_json", None):
        return item.b64_json, "image/png"
    if getattr(item, "url", None):
        fetched = httpx.get(item.url, timeout=60.0)
        fetched.raise_for_status()
        return base64.b64encode(fetched.content).decode("utf-8"), "image/png"
    raise Exception("OpenAI returned neither b64_json nor a url.")


@app.post("/generate-image")
def generate_image_endpoint(req: ImageRequest):
    """Returns one image as base64. The CREATOR APP uploads it to Supabase
    Storage through its existing asset path, so storage layout, draft_assets
    bookkeeping and RLS all stay in exactly one place."""
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Nothing to draw — this asset has no description yet.")
    if not req.api_key:
        raise HTTPException(status_code=400, detail="No image API key configured. Set one in Engine Config → Asset Art.")

    provider = (req.provider or "gemini").lower()
    model = (req.model_name or DEFAULT_IMAGE_MODELS.get(provider) or "").strip()
    prompt = _compose_image_prompt(req)

    try:
        _throttle(req.api_key)
        if provider == "gemini":
            image_b64, mime = _gemini_image(req.api_key, model, prompt, req.kind)
        elif provider == "openai":
            image_b64, mime = _openai_image(req.api_key, model, prompt, req.kind)
        else:
            raise HTTPException(status_code=400, detail=f"'{provider}' can't generate images. Use gemini or openai.")

        return {"status": "success", "image_base64": image_b64, "mime_type": mime,
                "model": model, "prompt_used": prompt}

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        kind = _classify_rate_limit(e)
        if kind == "daily":
            # "limit: 0" is a different animal from "you used your allowance":
            # the model is not on this account's tier at all, so waiting for
            # the daily reset achieves nothing. Say so, or the creator sits
            # there until midnight for a wall that never moves.
            never_allowed = re.search(r"limit:\s*0\b", str(e)) is not None
            if never_allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"'{model}' isn't available on this {provider} account's free tier "
                           f"(the quota for it is zero, not just used up — waiting won't help).\n\n"
                           f"👉 Enable billing on the key's project, or switch Engine Config → "
                           f"Asset Art to a provider whose key is on a paid plan. The Copy button "
                           f"on each tile still works for pasting into an external image tool."
                )
            raise HTTPException(
                status_code=429,
                detail=f"The image key has used up its daily quota on {provider}. "
                       f"Upload art manually for now, or switch the Asset Art provider in Engine Config."
            )
        if kind == "transient":
            wait = _extract_retry_after(str(e))
            raise HTTPException(
                status_code=429,
                detail=f"Image provider is rate-limiting"
                       + (f" — try again in about {int(wait)}s." if wait else " — try again shortly.")
            )
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


# ==========================================
# 7. EMAIL OTP / DEFERRED SIGNUP
# ==========================================
# The app no longer calls supabase.auth.signInWithOtp(). GoTrue has to INSERT
# an auth.users row before it has anything to hang a code on, which meant a
# stranger typing an address into the app created a user. Here the code sits in
# public.auth_otp_codes (hashed, TTL'd) and auth.users + public.profiles are
# both born in one shot, after the code checks out, via admin.create_user().
#
# The session comes from admin.generate_link(), which mints a one-shot token
# WITHOUT sending any email. The client trades that hash for a session through
# supabase.auth.verifyOtp({ token_hash }).

# Transactional mail goes out over the Brevo HTTPS API, not SMTP. Render's free
# tier drops outbound connections on ports 25/465/587, so smtplib worked on
# localhost and then timed out after 20s in production — a 502 that reads
# exactly like bad credentials. Port 443 is never blocked.
#
# Also retires the ~500/day consumer Gmail ceiling, which would have locked
# every user out for 24 hours on the first real traffic spike.
BREVO_API_KEY  = os.environ.get("BREVO_API_KEY")
BREVO_ENDPOINT = "https://api.brevo.com/v3/smtp/email"
SMTP_FROM      = os.environ.get("SMTP_FROM")
SMTP_FROM_NAME = os.environ.get("SMTP_FROM_NAME", "Vystoria")


# Peppering means a dump of auth_otp_codes alone can't be brute-forced offline
# for the 10^6 possible codes.
OTP_PEPPER         = os.environ.get("OTP_PEPPER") or SUPABASE_SERVICE_KEY
OTP_TTL_MINUTES    = int(os.environ.get("OTP_TTL_MINUTES", "10"))
OTP_RESEND_SECONDS = int(os.environ.get("OTP_RESEND_SECONDS", "60"))
OTP_MAX_ATTEMPTS   = int(os.environ.get("OTP_MAX_ATTEMPTS", "5"))

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class OtpRequest(BaseModel):
    email: str


class OtpVerify(BaseModel):
    email: str
    code: str


def _normalize_email(raw: str) -> str:
    email = (raw or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="Please enter a valid email address.")
    return email


def _hash_otp(email: str, code: str) -> str:
    return hashlib.sha256(f"{OTP_PEPPER}:{email}:{code}".encode("utf-8")).hexdigest()


def _lookup_profile(email: str):
    """public.profiles is the app's definition of 'registered', and
    onboarded_at is what separates a real account from a Google OAuth row whose
    owner backed out of the "New to Vystoria?" card. A row with onboarded_at
    IS NULL has to read as a first-time address here, or the verify screen
    greets an abandoned signup with "Welcome Back" and request-otp reports the
    wrong is_new_user."""
    res = (
        supabase.table("profiles")
        .select("id, full_name, onboarded_at")
        .eq("email", email)
        .not_.is_("onboarded_at", "null")
        .limit(1)
        .execute()
    )
    rows = res.data or []
    return rows[0] if rows else None


def _mark_onboarded(email: str) -> None:
    """Stamps intent. Idempotent — the .is_(null) filter makes it a no-op for
    an account that is already onboarded, so it can be called unconditionally.

    Entering a correct code IS the confirmation of intent on the OTP path, so
    there is no second card to tap. The profiles row is born NULL because the
    on_auth_user_confirmed trigger fires inside admin.create_user() and knows
    nothing about which path got us here."""
    try:
        (
            supabase.table("profiles")
            .update({"onboarded_at": datetime.now(timezone.utc).isoformat()})
            .eq("email", email)
            .is_("onboarded_at", "null")
            .execute()
        )
    except Exception:
        # Never fail a valid login over a UX flag. Worst case the user sees the
        # new-account card once more and Confirm re-stamps it.
        traceback.print_exc()


def _send_otp_email(email: str, code: str) -> None:
    """Delivered over HTTPS rather than SMTP — see the note above the config
    block. Raises on any non-2xx so request_otp_endpoint can delete the code it
    just wrote: a live code with no email behind it is worse than a clean
    failure, because the user has no way to ever satisfy it."""
    if not BREVO_API_KEY:
        raise RuntimeError("BREVO_API_KEY is not configured on the server.")
    if not SMTP_FROM:
        raise RuntimeError("SMTP_FROM is not configured on the server.")

    payload = {
        "sender": {"email": SMTP_FROM, "name": SMTP_FROM_NAME},
        "to": [{"email": email}],
        "subject": f"{code} is your Vystoria verification code",
        "textContent": (
            f"Your Vystoria verification code is {code}.\n\n"
            f"It expires in {OTP_TTL_MINUTES} minutes. If you didn't ask for it, ignore this email.\n"
        ),
        "htmlContent": (
            f"""<html><body style="font-family:Manrope,Arial,sans-serif;background:#0B0B14;padding:32px;color:#fff">
              <h2 style="font-family:Georgia,serif;color:#fff;margin:0 0 12px">Vystoria</h2>
              <p style="color:#C2BBD4;margin:0 0 20px">Here is your verification code.</p>
              <p style="font-size:34px;letter-spacing:10px;font-weight:700;color:#C48DFF;margin:0 0 20px">{code}</p>
              <p style="color:#B0A9C4;font-size:13px;margin:0">
                It expires in {OTP_TTL_MINUTES} minutes. If you didn't request it, you can ignore this email.
              </p>
            </body></html>"""
        ),
    }

    # 15s: well inside the client's 45s budget, and far below the 20s smtplib
    # timeout that was eating the whole request.
    resp = httpx.post(
        BREVO_ENDPOINT,
        headers={"api-key": BREVO_API_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=15.0,
    )

    if resp.status_code >= 300:
        # Surfaced by the traceback.print_exc() that request_otp_endpoint
        # already runs before raising its 502.
        raise RuntimeError(f"Brevo rejected the send ({resp.status_code}): {resp.text}")

def _mint_session_token(email: str) -> str:
    """Admin generate_link returns a one-shot hashed token and does NOT send an
    email. The client exchanges it for a real session."""
    res = supabase.auth.admin.generate_link({"type": "magiclink", "email": email})
    props = getattr(res, "properties", None)
    if props is None and isinstance(res, dict):
        props = res.get("properties")

    hashed = getattr(props, "hashed_token", None)
    if hashed is None and isinstance(props, dict):
        hashed = props.get("hashed_token")

    if not hashed:
        raise HTTPException(status_code=500, detail="Could not start your session. Please try again.")
    return hashed


@app.post("/auth/request-otp")
def request_otp_endpoint(req: OtpRequest):
    """Issues a code. Writes NOTHING to auth.users or public.profiles."""
    email = _normalize_email(req.email)
    now = datetime.now(timezone.utc)

    try:
        supabase.table("auth_otp_codes").delete().lt("expires_at", now.isoformat()).execute()

        existing = (
            supabase.table("auth_otp_codes")
            .select("last_sent_at")
            .eq("email", email)
            .limit(1)
            .execute()
        ).data or []

        if existing and existing[0].get("last_sent_at"):
            last_sent = datetime.fromisoformat(existing[0]["last_sent_at"].replace("Z", "+00:00"))
            waited = (now - last_sent).total_seconds()
            if waited < OTP_RESEND_SECONDS:
                raise HTTPException(
                    status_code=429,
                    detail=f"Please wait {int(OTP_RESEND_SECONDS - waited)}s before requesting another code.",
                )

        profile = _lookup_profile(email)
        code = f"{secrets.randbelow(1_000_000):06d}"

        supabase.table("auth_otp_codes").upsert(
            {
                "email": email,
                "code_hash": _hash_otp(email, code),
                "expires_at": (now + timedelta(minutes=OTP_TTL_MINUTES)).isoformat(),
                "attempts": 0,
                "last_sent_at": now.isoformat(),
            },
            on_conflict="email",
        ).execute()

        try:
            _send_otp_email(email, code)
        except Exception:
            traceback.print_exc()
            # Don't leave a live code behind for a mail that never went out.
            supabase.table("auth_otp_codes").delete().eq("email", email).execute()
            raise HTTPException(status_code=502, detail="We couldn't send the code. Please try again in a moment.")

        # This is what lets the verify screen say the right thing before the
        # user types anything — no guessing from client-side metadata.
        return {
            "is_new_user": profile is None,
            "display_name": None if profile is None else (profile.get("full_name") or ""),
        }
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Could not send the code: {str(e)}")


@app.post("/auth/verify-otp")
def verify_otp_endpoint(req: OtpVerify):
    """Checks the code and only THEN creates the account."""
    email = _normalize_email(req.email)
    code = (req.code or "").strip()
    if not code.isdigit() or len(code) != 6:
        raise HTTPException(status_code=400, detail="Enter the 6-digit code from your email.")

    try:
        rows = (
            supabase.table("auth_otp_codes")
            .select("code_hash, expires_at, attempts")
            .eq("email", email)
            .limit(1)
            .execute()
        ).data or []

        if not rows:
            raise HTTPException(status_code=400, detail="That code has expired. Please request a new one.")

        row = rows[0]
        expires_at = datetime.fromisoformat(row["expires_at"].replace("Z", "+00:00"))
        if datetime.now(timezone.utc) > expires_at:
            supabase.table("auth_otp_codes").delete().eq("email", email).execute()
            raise HTTPException(status_code=400, detail="That code has expired. Please request a new one.")

        if int(row.get("attempts") or 0) >= OTP_MAX_ATTEMPTS:
            supabase.table("auth_otp_codes").delete().eq("email", email).execute()
            raise HTTPException(status_code=429, detail="Too many wrong attempts. Please request a new code.")

        if not hmac.compare_digest(row["code_hash"], _hash_otp(email, code)):
            supabase.table("auth_otp_codes").update(
                {"attempts": int(row.get("attempts") or 0) + 1}
            ).eq("email", email).execute()
            raise HTTPException(status_code=400, detail="That code isn't right. Please check and try again.")

        # Correct — burn it immediately so it can't be replayed.
        supabase.table("auth_otp_codes").delete().eq("email", email).execute()

        profile = _lookup_profile(email)
        is_new_user = profile is None

        if is_new_user:
            try:
                # email_confirm=True means the row lands already-confirmed, so
                # the profiles trigger fires on INSERT — the same path Google
                # OAuth users take. full_name seeds the welcome screen with
                # something better than the 'Player One' placeholder.
                supabase.auth.admin.create_user(
                    {
                        "email": email,
                        "email_confirm": True,
                        "user_metadata": {"full_name": email.split("@")[0]},
                    }
                )
            except Exception as create_error:
                # "already registered" now covers a real case, not just legacy
                # junk: the address has an abandoned Google OAuth row sitting at
                # onboarded_at IS NULL. Adopting it is correct — the human is
                # proving ownership of the same mailbox right now.
                if "already" not in str(create_error).lower():
                    traceback.print_exc()
                    raise HTTPException(status_code=500, detail="Could not finish setting up your account.")

        # Unconditional: heals any straggler (an abandoned Google row just
        # adopted above, or a legacy account predating the column) and is a
        # no-op for anyone already stamped.
        _mark_onboarded(email)

        return {"token_hash": _mint_session_token(email), "is_new_user": is_new_user}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Verification failed: {str(e)}")        