import os
import json
import re
import time
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

# ==========================================
# 2. DATA MODELS
# ==========================================
class GenerateRequest(BaseModel):
    provider: str      # 'gemini', 'openai', 'grok', 'claude'
    api_key: str       # Custom key provided by the creator
    model_name: str    # e.g., 'gemini-1.5-flash', 'gpt-4o', 'grok-2-beta'
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
4. **Core Rules & Systems** 5. **Major Themes** and **Emotional Arc**
6. **Writing Style Guidelines**
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

Write Chapter {chapter_number}. Generate 15-25 scenes.
Give the player a meaningful choice roughly every 3-5 scenes — do not let more than 5
scenes pass in a row without a branching choice. Aim for at least 4-6 choice points
across the chapter, not just 1-2. Choices should meaningfully diverge (different next_scene
paths), not just reword the same outcome.
For scenes with NO choices, use "next_scene_default": "next_scene_id".

**CHOICE FORMATTING RULES (very important):**
1. Every scene that has "choices" MUST also include a "choice_prompt" field: 1-2 sentences
   of in-world narrative or dialogue text that sets up the decision — e.g. a question a
   character asks, a moment of tension, or what the protagonist is weighing. This text
   appears on screen right before the choice buttons, so it must give the player real
   context for what they're deciding. NEVER use generic filler like "Make a decision" or
   "What will you do" — write the actual situational text.
2. Each individual choice's "text" MUST be SHORT — a maximum of 6-7 words. Write it as a
   punchy action or phrase, not a full sentence. 
   Good: "Fight the guard", "Ask about the ring", "Stay silent", "Trust her warning"
   Bad: "You decide to attack the guard before he can call for backup"

**CRITICAL JSON RULES:**
1. Output ONLY valid JSON. No conversational text before or after.
2. If you use quotes inside "text", you MUST escape them. Example: "text": "She said, \\"Hello.\\""
3. Do NOT include trailing commas at the end of lists or objects.
4. The FINAL scene of the chapter MUST NOT have choices. It must end linearly using "next_scene_default".

**Output ONLY valid JSON** in this exact structure:
{{
  "chapter_number": {chapter_number},
  "chapter_title": "...",
  "scenes": [
    {{
      "id": "ch{chapter_number}_scene01",
      "background": "black",
      "next_scene_default": "ch{chapter_number}_scene02", 
      "sequence": [
        {{ "type": "narrative", "text": "..." }},
        {{ "type": "dialogue", "speaker": "Name", "text": "..." }}
      ],
      "choice_prompt": "The guard's hand moves to his sword. There's no more time to think.",
      "choices": [
        {{ "text": "Fight the guard", "next_scene": "ch{chapter_number}_scene02a" }},
        {{ "text": "Try to talk him down", "next_scene": "ch{chapter_number}_scene02b" }}
      ]
    }}
  ]
}}

Note: scenes with NO choices should omit "choice_prompt" and "choices" entirely and just use "next_scene_default".
"""

ASSET_MANIFEST_PROMPT = """
You are cataloging the visual assets needed for a Visual Novel.

**World Bible:**
{world_bible}

**Speaker names that actually appear in the finished story (one entry each, no more/fewer):**
{speakers}

**Background location IDs that actually appear (one entry each, no more/fewer):**
{backgrounds}

For each speaker, write a concise (1-2 sentence) physical/costume description for a portrait artist.
For each background ID, write a concise (1-2 sentence) setting description for a background artist.
Also write ONE cover art description: a striking, poster-style scene (1-2 sentences) that
captures the story's tone and would work as a thumbnail/cover image.

Output ONLY valid JSON in this exact structure:
{{
  "characters": [ {{ "name": "...", "description": "..." }} ],
  "backgrounds": [ {{ "id": "...", "description": "..." }} ],
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
   established in the World Bible? Flag any location/power/character inconsistency.

C. Stylistic & Tonal Cohesion — Does the prose match the requested genre/tone and the World
   Bible's writing style guidelines? Flag generic tropes or tone-breaking modern slang.

D. Character Voice & Agency — Do characters have distinct speech patterns matching their
   profiles? Does the protagonist make active decisions rather than passively drifting?

E. Interactive UX & Narrative Flow — Is the narration -> dialogue -> choice_prompt -> choices
   pacing natural? Flag jarring scene transitions or weak/generic choice prompts.

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

# NEW: targeted single-scene rewrite prompt. Deliberately scoped so the model
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
3. You MAY change: background, sequence (narrative/dialogue text), choice_prompt, and the wording
   of each choice's "text" — but if choices exist, the NUMBER of choices and their "next_scene"
   targets must stay the same as the original unless the creator's instruction explicitly asks you
   to add or remove a branch.
4. Keep the same choice-formatting rules as the rest of the story: choice_prompt required whenever
   choices exist (1-2 sentences of real in-world setup, never generic filler), and each choice
   "text" is a punchy phrase under ~7 words.

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

# ==========================================
# 4. MULTI-MODEL ADAPTER & PARSER
# ==========================================
def call_llm(prompt, system_instruction, provider, api_key, model_name):
    """Dynamically routes the prompt to the selected LLM provider."""
    provider = provider.lower()

    if provider == 'gemini':
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or 'gemini-1.5-flash', system_instruction=system_instruction)
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
            model=model_name or "gpt-4o",
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    elif provider == 'grok':
        import openai
        client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        resp = client.chat.completions.create(
            model=model_name or "grok-2-beta",
            messages=[{"role": "system", "content": system_instruction}, {"role": "user", "content": prompt}]
        )
        return resp.choices[0].message.content

    elif provider == 'claude':
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=model_name or "claude-3-5-sonnet-20240620",
            max_tokens=4000,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt}]
        )
        return resp.content[0].text

    else:
        raise ValueError(f"Unsupported Provider: {provider}")

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

def build_asset_manifest(world_bible, speaker_names, background_ids, provider, api_key, model_name):
    """Catalogs character/background/cover art descriptions. Never raises —
    falls back to bare names if the LLM call or JSON parse fails, so this can
    safely run inside a thread pool alongside the Judge evaluation."""
    try:
        manifest_raw = call_llm(
            ASSET_MANIFEST_PROMPT.format(
                world_bible=world_bible,
                speakers="\n".join(f"- {s}" for s in speaker_names) or "(none)",
                backgrounds="\n".join(f"- {b}" for b in background_ids) or "(none)"
            ),
            "Output ONLY valid JSON.", provider, api_key, model_name
        )
        asset_manifest = json.loads(clean_json_output(manifest_raw))
        described_chars = {c.get('name') for c in asset_manifest.get('characters', [])}
        described_bgs = {b.get('id') for b in asset_manifest.get('backgrounds', [])}
        for name in speaker_names:
            if name not in described_chars:
                asset_manifest.setdefault('characters', []).append({"name": name, "description": ""})
        for bg in background_ids:
            if bg not in described_bgs:
                asset_manifest.setdefault('backgrounds', []).append({"id": bg, "description": ""})
        return asset_manifest, None
    except Exception as e:
        fallback = {
            "characters": [{"name": n, "description": ""} for n in speaker_names],
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
    """Rewrites ONE scene per a targeted creator instruction. Raises on
    failure — the caller (endpoint) is responsible for turning that into an
    HTTP error, since (unlike the judge/asset helpers) there's no safe
    fallback for "the scene the creator asked to fix" other than the
    unmodified original, which the frontend already has."""
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

    # Guardrails: never let the model drift the id or drop/relink to a scene
    # id that didn't already exist on this scene, since that would silently
    # break navigation elsewhere in the story.
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
                # Model invented a new id — refuse rather than ship a dead link.
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

        update_task('generating', 'Building World Bible...', 5, f"Started {req.provider.upper()} engine. Chapters targeted: {num_chapters}")

        idea_section = (
            f"- Core Idea (build the plot and world firmly around this creator-provided concept): {req.idea}"
            if req.idea and req.idea.strip() else ""
        )
        idea_reminder = (
            f"- Stay faithful to this core idea from the creator: {req.idea}"
            if req.idea and req.idea.strip() else ""
        )

        # NEW: reference-document ingestion. If the creator attached a draft,
        # outline, or lore doc, the World/Outline prompts switch into "adapt
        # this faithfully" mode instead of inventing a fresh plot.
        has_reference = bool(req.reference_text and req.reference_text.strip())
        trimmed_reference = (req.reference_text or "").strip()[:MAX_REFERENCE_CHARS]
        reference_section = (
            f"- A Reference Document has been provided below. Treat it as the AUTHORITATIVE source: "
            f"adapt its plot, characters, and setting faithfully into the World Bible format rather "
            f"than inventing a different story. Only invent details necessary to fill gaps (minor "
            f"side characters, extra locations) while staying fully consistent with the document.\n\n"
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
                                 idea_section=idea_section, reference_section=reference_section),
            "You are a master visual novel author.", req.provider, req.api_key, req.model_name
        )

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
                chapter_number=i, world_bible=world_bible, outline=outline, previous_summary=previous_summary
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
                except Exception as e:
                    update_task('generating', step_msg, base_prog, f"⚠️ Attempt {attempt + 1} Failed (JSON error). Retrying...")
                    time.sleep(2)

            if not scenes:
                raise Exception(f"Failed to generate valid JSON for Chapter {i} after {MAX_RETRIES} attempts.")

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

        speaker_names = sorted({
            block.get('speaker') for scene in all_scenes for block in scene.get('sequence', [])
            if block.get('speaker')
        })
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
                    "Scanning story for unique speakers/settings, and starting the AI Quality Judge in parallel...")

        with ThreadPoolExecutor(max_workers=2) as executor:
            assets_future = executor.submit(
                build_asset_manifest, world_bible, speaker_names, background_ids,
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
                        f"⚠️ Could not auto-describe assets ({asset_err}); showing names only.")
        else:
            update_task('generating', 'Asset manifest ready.', 92,
                        f"Cataloged {len(speaker_names)} characters and {len(background_ids)} backgrounds.")

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
    """
    Rewrites exactly one scene per the creator's instruction. Does not touch
    Supabase at all — the frontend hot-swaps the returned scene into its own
    in-memory story state, and the full story is only persisted again if/when
    the creator explicitly saves a draft.
    """
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
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Scene tweak failed: {str(e)}")