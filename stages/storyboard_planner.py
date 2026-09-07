"""
stages/storyboard_planner.py — Stage 2: Generate scene-by-scene storyboard via Ollama LLM.

Takes deep_analysis.json and produces a storyboard with 50-80 scenes,
each with narration text, visual descriptions, Manim animation intent,
and timing information.

The storyboard is grouped into logical sections (history, concepts, examples, etc.)
and follows a pedagogical arc designed to build understanding progressively.
"""

import json
import re
from pathlib import Path

import ollama

from config import (
    OLLAMA_BASE_URL,
    STORYBOARD_MODEL,
    STORYBOARD_TEMPERATURE,
    TARGET_MIN_MINUTES,
    TARGET_MAX_MINUTES,
    DEFAULT_SCENE_DURATION,
    calc_target_scenes,
)


STORYBOARD_SYSTEM = """You are an award-winning educational video director who creates 3Blue1Brown-style math videos.

You design DETAILED scene-by-scene storyboards for Manim-animated educational videos.
Each scene is a self-contained Manim animation with specific objects, transformations, and timing.

Your videos are:
  - Visually stunning with smooth animations and clean typography
  - Pedagogically structured: build intuition before formalism
  - Engaging: use stories, analogies, and "aha!" moments
  - Comprehensive: cover everything a student needs to master the topic
  - 30-60 minutes long with ~30 seconds per scene

CRITICAL: You must output ONLY valid JSON. No markdown, no commentary."""


STORYBOARD_PROMPT = """Create a comprehensive Manim-animated video storyboard for this chapter.

CHAPTER: {chapter_title}
TARGET DURATION: {target_min}-{target_max} minutes ({target_scenes} scenes × ~{scene_duration}s each)

DEEP ANALYSIS (use ALL of this):
{analysis_json}

CHAPTER FLOW ORDER:
{chapter_flow}

Return this EXACT JSON structure:
{{
    "title": "Video title — engaging, descriptive (e.g. 'Sets: From Cantor's Paradox to Your Exam Paper')",
    "description": "2-3 sentence video description for YouTube",
    "total_duration_estimate_minutes": 45,

    "sections": [
        {{
            "section_id": 1,
            "section_title": "The Story of Sets — Where It All Began",
            "section_type": "HISTORY | CONCEPT | EXAMPLES | TRICKS | FUN_FACTS | SUMMARY",
            "duration_minutes": 5,
            "scenes": [
                {{
                    "scene_id": 1,
                    "title": "Scene title (short, descriptive)",
                    "duration_seconds": 30,
                    "teaching_beat": "HOOK | HISTORY | DEFINITION | INTUITION | EXAMPLE | TRICK | FUN_FACT | MISCONCEPTION | PRACTICE | SUMMARY | TRANSITION",

                    "narration_text": "Full narration script for this scene (40-80 words). Conversational, warm, like 3Blue1Brown. Use analogies. This IS the voiceover.",

                    "visual_description": "Detailed description of what the viewer SEES on screen (60-100 words). Describe animations, objects appearing/disappearing, transformations, colors, positions.",

                    "manim_intent": "Specific Manim instructions: what objects to create, what animations to use (Create, FadeIn, Transform, Write, MoveToTarget, etc.), layout/positioning, colors. Be PRECISE about Manim API.",

                    "key_objects": ["Text", "MathTex", "Circle", "VGroup", "Arrow", "NumberLine", "Rectangle", "Ellipse", "Table"],

                    "background_color": "#1a1a2e",
                    "accent_colors": ["#e94560", "#0f3460", "#16213e"],

                    "transition_in": "FadeIn | GrowFromCenter | Write | Create",
                    "transition_out": "FadeOut | ShrinkToCenter | Uncreate",

                    "on_screen_text": [
                        "Any text/equations that appear on screen (LaTeX format for math)"
                    ],

                    "notes": "Any special instructions for this scene"
                }}
            ]
        }}
    ],

    "color_scheme": {{
        "background": "#1a1a2e",
        "primary_text": "#ffffff",
        "accent_1": "#e94560",
        "accent_2": "#0f3460",
        "accent_3": "#16213e",
        "highlight": "#ffd700",
        "success": "#00ff88",
        "error": "#ff4444"
    }},

    "recurring_elements": {{
        "title_style": "Description of how section titles appear",
        "definition_style": "How definitions are highlighted (box, underline, glow)",
        "example_style": "How worked examples are presented",
        "transition_style": "How scenes transition between sections"
    }}
}}

SECTION PLAN — follow this order:
1. HOOK (1-2 scenes): Jaw-dropping visual or paradox to grab attention
2. HISTORY (0-1 scenes): Brief origin. Use the ancient_history data sparingly to avoid making the video too history-heavy.
3. CORE CONCEPTS (15-25 scenes): Each concept from core_concepts gets 1-3 scenes.
   - First show intuition/analogy
   - Then formal definition
   - Then a quick example
4. MISCONCEPTIONS (3-5 scenes): Address each misconception with visual proof of why it's wrong
5. FUN FACTS (3-5 scenes): Mind-blowing connections
6. TRICKS & SHORTCUTS (4-6 scenes): Exam-useful techniques with examples
7. WORKED EXAMPLES (8-12 scenes): Step-by-step animated solutions
8. SUMMARY & RECAP (2-3 scenes): Key takeaways, visual summary card

CRITICAL RULES:
- EVERY scene must have specific manim_intent with real Manim class names
- narration_text is the COMPLETE voiceover script — not a summary
- visual_description describes what the VIEWER SEES, not what to code
- Use smooth, beautiful color scheme throughout (dark background, vibrant accents)
- Include ALL concepts from the analysis — don't skip any
- Each scene ~30 seconds of content
- Total should be {target_scenes} scenes across all sections
- Think like 3Blue1Brown: clean, elegant, mathematically beautiful"""


def plan_storyboard(
    deep_analysis: dict,
    output_path: str = None,
    model: str = None,
    target_minutes: int = 45,
) -> dict:
    """
    Generate a comprehensive scene-by-scene storyboard.

    Args:
        deep_analysis: Output from deep_analyzer.analyze_chapter()
        output_path: Save storyboard JSON here
        model: Override storyboard model
        target_minutes: Target video duration in minutes

    Returns:
        Storyboard dict with sections and scenes
    """
    model = model or STORYBOARD_MODEL
    target_scenes = calc_target_scenes(target_minutes)

    # Trim analysis to fit context window
    analysis_trimmed = _trim_analysis(deep_analysis)

    chapter_flow = "\n".join(
        f"  {i+1}. {step}"
        for i, step in enumerate(deep_analysis.get("chapter_flow", ["Full chapter"]))
    )

    prompt = STORYBOARD_PROMPT.format(
        chapter_title=deep_analysis.get("chapter_title", "Unknown Chapter"),
        target_min=TARGET_MIN_MINUTES,
        target_max=TARGET_MAX_MINUTES,
        target_scenes=target_scenes,
        scene_duration=DEFAULT_SCENE_DURATION,
        analysis_json=json.dumps(analysis_trimmed, indent=2, ensure_ascii=False),
        chapter_flow=chapter_flow,
    )

    print(f"[Storyboard] Generating {target_scenes}-scene storyboard with {model}...")
    print(f"[Storyboard] Target: {target_minutes} min video")
    print(f"[Storyboard] This may take 5-10 minutes for a detailed storyboard...")

    client = ollama.Client(host=OLLAMA_BASE_URL)

    messages = [
        {"role": "system", "content": STORYBOARD_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        response = client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": STORYBOARD_TEMPERATURE,
                "num_predict": 32768,  # Long response needed for 50+ scenes
            },
            format="json",
        )

        text = response.get("message", {}).get("content", "").strip()

        try:
            storyboard = _robust_json_parse(text)
            _validate_storyboard(storyboard)
            break
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                raise RuntimeError(f"Failed to generate valid storyboard after 3 attempts: {e}")
            print(f"[Storyboard] Warning: LLM produced invalid JSON ({e}). Asking it to self-correct... (Attempt {attempt+1}/3)")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Your previous output was invalid JSON or failed validation: {e}. Please fix the formatting/content and return ONLY valid JSON."})

    # Assign global scene IDs
    global_id = 1
    for section in storyboard.get("sections", []):
        for scene in section.get("scenes", []):
            scene["global_scene_id"] = global_id
            global_id += 1

    storyboard["total_scenes"] = global_id - 1

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(storyboard, f, indent=2, ensure_ascii=False)
        print(f"[Storyboard] Saved -> {output_path}")

    # Also export human-readable markdown
    md_path = str(Path(output_path).parent / "STORYBOARD.md") if output_path else None
    if md_path:
        _export_storyboard_md(storyboard, md_path)
        print(f"[Storyboard] Readable version -> {md_path}")

    _print_summary(storyboard)
    return storyboard


def _trim_analysis(analysis: dict) -> dict:
    """Trim analysis to fit within context window while keeping key info."""
    trimmed = {}
    for key, value in analysis.items():
        if key == "full_text":
            continue  # Skip raw text
        if isinstance(value, list):
            # Keep all items but trim long content fields
            trimmed_items = []
            for item in value:
                if isinstance(item, dict):
                    trimmed_item = {}
                    for k, v in item.items():
                        if isinstance(v, str) and len(v) > 500:
                            trimmed_item[k] = v[:500] + "..."
                        elif isinstance(v, list) and len(v) > 10:
                            trimmed_item[k] = v[:10]
                        else:
                            trimmed_item[k] = v
                    trimmed_items.append(trimmed_item)
                else:
                    trimmed_items.append(item)
            trimmed[key] = trimmed_items
        else:
            trimmed[key] = value

    return trimmed


def _validate_storyboard(storyboard: dict) -> None:
    """Validate the storyboard has required structure."""
    if "sections" not in storyboard:
        raise ValueError("Storyboard missing 'sections' key")

    total_scenes = sum(
        len(s.get("scenes", []))
        for s in storyboard["sections"]
    )

    if total_scenes < 10:
        raise ValueError(f"Storyboard has only {total_scenes} scenes — expected at least 20")

    print(f"[Storyboard] Validation passed: {total_scenes} scenes across {len(storyboard['sections'])} sections")


def _export_storyboard_md(storyboard: dict, output_path: str) -> None:
    """Export storyboard as human-readable Markdown."""
    lines = []
    lines.append(f"# {storyboard.get('title', 'Educational Video Storyboard')}")
    lines.append("")
    lines.append(f"**Description:** {storyboard.get('description', '')}")
    lines.append(f"**Estimated Duration:** {storyboard.get('total_duration_estimate_minutes', '?')} minutes")
    lines.append(f"**Total Scenes:** {storyboard.get('total_scenes', '?')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    for section in storyboard.get("sections", []):
        lines.append(f"## Section {section.get('section_id', '?')}: {section.get('section_title', 'Untitled')}")
        lines.append(f"*Type: {section.get('section_type', '?')} | Duration: ~{section.get('duration_minutes', '?')} min*")
        lines.append("")

        for scene in section.get("scenes", []):
            sid = scene.get("global_scene_id", scene.get("scene_id", "?"))
            lines.append(f"### Scene {sid}: {scene.get('title', 'Untitled')}")
            lines.append(f"*{scene.get('duration_seconds', 30)}s | Beat: {scene.get('teaching_beat', '?')}*")
            lines.append("")
            lines.append(f"**🎙️ Narration:**")
            lines.append(f"> {scene.get('narration_text', '(no narration)')}")
            lines.append("")
            lines.append(f"**👁️ Visual:**")
            lines.append(f"> {scene.get('visual_description', '(no visual description)')}")
            lines.append("")
            lines.append(f"**🎬 Manim Intent:**")
            lines.append(f"> {scene.get('manim_intent', '(no manim intent)')}")
            lines.append("")

            on_screen = scene.get("on_screen_text", [])
            if on_screen:
                lines.append(f"**📝 On Screen:** {' | '.join(on_screen)}")
                lines.append("")

            lines.append("---")
            lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _print_summary(storyboard: dict) -> None:
    """Print storyboard summary."""
    print(f"\n{'='*60}")
    print(f"  Storyboard: {storyboard.get('title', 'Untitled')}")
    print(f"{'='*60}")
    print(f"  Duration: ~{storyboard.get('total_duration_estimate_minutes', '?')} min")
    print(f"  Scenes:   {storyboard.get('total_scenes', '?')}")
    print(f"  Sections: {len(storyboard.get('sections', []))}")
    print(f"{'='*60}")

    for section in storyboard.get("sections", []):
        scene_count = len(section.get("scenes", []))
        print(f"  [{section.get('section_id', '?'):02d}] {section.get('section_title', '?'):40s} ({scene_count} scenes, ~{section.get('duration_minutes', '?')} min)")

    print(f"{'='*60}\n")


def _robust_json_parse(text: str) -> dict:
    """
    Robustly parse JSON generated by LLMs, fixing common syntax issues:
    - Missing commas between fields/objects
    - Unescaped newlines inside strings
    - Truncated JSON near the end
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # Attempt 1: Standard json.loads
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Fix missing commas between properties/objects
    fixed = re.sub(r'("\s*|\d+|true|false|null|\]|\})\s*\n\s*("|\{|\[)', r'\1,\n\2', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: Fix trailing commas
    fixed = re.sub(r',\s*([\}\]])', r'\1', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 4: Truncation repair — balance brackets and braces
    open_brackets = 0
    open_braces = 0
    in_string = False
    escape = False

    cleaned_chars = []
    for char in fixed:
        cleaned_chars.append(char)
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if not in_string:
            if char == '{':
                open_braces += 1
            elif char == '}':
                open_braces -= 1
            elif char == '[':
                open_brackets += 1
            elif char == ']':
                open_brackets -= 1

    if in_string:
        cleaned_chars.append('"')

    res_str = "".join(cleaned_chars).rstrip()
    if res_str.endswith(','):
        res_str = res_str[:-1]

    res_str += ']' * max(0, open_brackets)
    res_str += '}' * max(0, open_braces)

    try:
        return json.loads(res_str)
    except json.JSONDecodeError:
        pass

    # Final fallback: Regex extract individual scene objects
    scene_matches = re.findall(r'\{\s*"scene_id".*?\}(?=\s*,|\s*\]|\s*\}|\Z)', text, re.DOTALL)
    recovered_scenes = []
    for sc in scene_matches:
        try:
            recovered_scenes.append(json.loads(sc))
        except Exception:
            pass

    if recovered_scenes:
        return {
            "title": "Educational Video Storyboard",
            "description": "Auto-recovered storyboard",
            "total_duration_estimate_minutes": 45,
            "sections": [{"section_id": 1, "section_title": "Chapter Content", "scenes": recovered_scenes}]
        }

    raise json.JSONDecodeError(f"Failed to parse LLM JSON (len {len(text)})", text, 0)
