"""
stages/deep_analyzer.py — Stage 1: Deep concept extraction via Ollama LLM.

Takes the extracted chapter text and produces a comprehensive analysis covering:
  - Ancient history & origins of the topic
  - Core concepts & definitions
  - Common misconceptions
  - Fun facts & surprising connections
  - Tricks & shortcuts for exams
  - Worked examples with step-by-step solutions
  - Visual animation ideas for each concept
"""

import json
import re
from pathlib import Path

import ollama

from config import OLLAMA_BASE_URL, STORYBOARD_MODEL, STORYBOARD_TEMPERATURE


DEEP_ANALYSIS_SYSTEM = """You are an expert educational content analyst specializing in Indian school curricula (NCERT, CBSE).
You have deep knowledge of the history of mathematics, pedagogy, and how to make abstract concepts engaging.

Your task: Analyze the provided chapter text and extract EVERYTHING a student needs to deeply understand this topic.
Go FAR beyond what the textbook says — add historical context, surprising connections, exam tricks, and fun facts.

You must output ONLY valid JSON. No markdown fences, no preamble, no extra text."""


DEEP_ANALYSIS_PROMPT = """Analyze this NCERT chapter thoroughly and return a comprehensive educational breakdown.

CHAPTER TEXT:
{chapter_text}

SECTIONS FOUND:
{sections_summary}

DEFINITIONS FOUND:
{definitions_summary}

EXAMPLES FOUND:
{examples_summary}

Return this EXACT JSON structure:
{{
    "chapter_title": "Full chapter title",
    "subject": "Mathematics",
    "grade": "Class 11",
    "topic_summary": "2-3 sentence overview of the entire chapter",

    "ancient_history": [
        {{
            "era": "e.g. 1874 / Ancient Greece / 17th Century",
            "person": "Name of mathematician/scholar",
            "event": "What happened and why it matters (2-3 sentences)",
            "fun_angle": "Why a student would find this interesting"
        }}
    ],

    "core_concepts": [
        {{
            "name": "Concept name (e.g. 'Roster Form', 'Union of Sets')",
            "definition": "Clear, precise definition in simple language",
            "notation": "Mathematical notation if applicable",
            "intuition": "Everyday analogy or intuitive explanation",
            "visual_idea": "How to animate this in Manim (specific objects, transformations)",
            "importance": "Why this concept matters / where it's used"
        }}
    ],

    "misconceptions": [
        {{
            "wrong_belief": "What students commonly get wrong",
            "why_wrong": "Why it seems right but isn't",
            "correct_understanding": "The right way to think about it",
            "visual_fix": "How an animation could demonstrate the correct idea"
        }}
    ],

    "fun_facts": [
        {{
            "fact": "The surprising fact",
            "connection": "How it connects to the chapter topic",
            "wow_factor": "Why this is mind-blowing"
        }}
    ],

    "tricks_and_shortcuts": [
        {{
            "name": "Short name for the trick",
            "description": "How to use this trick",
            "when_to_use": "In what exam/problem situations",
            "example": "Quick example showing the trick in action"
        }}
    ],

    "worked_examples": [
        {{
            "problem": "Problem statement",
            "category": "e.g. Roster to Set-Builder, Union, Complement",
            "difficulty": "easy | medium | hard",
            "solution_steps": [
                "Step 1: ...",
                "Step 2: ...",
                "Step 3: ..."
            ],
            "answer": "Final answer",
            "visual_idea": "How to animate this solution step-by-step"
        }}
    ],

    "chapter_flow": [
        "Ordered list of topics as they should be taught for maximum understanding"
    ],

    "exam_weightage": "How important this chapter is for boards/JEE and which types of questions appear"
}}

CRITICAL INSTRUCTIONS:
- For ancient_history: Include AT MOST 1 brief entry for historical context. Do not over-focus on history.
- For core_concepts: Cover EVERY concept in the chapter — don't skip any
- For misconceptions: Include at least 5 common student errors
- For fun_facts: Include at least 5 genuinely surprising facts
- For tricks_and_shortcuts: Include at least 4 exam-useful tricks
- For worked_examples: Include at least 8 examples of varying difficulty
- For visual_idea fields: Be SPECIFIC about Manim objects (Circle, VGroup, MathTex, NumberLine, Arrow, etc.)
- Make everything accessible to a Class 11 student but intellectually stimulating"""


def analyze_chapter(
    chapter_content: dict,
    output_path: str = None,
    model: str = None,
) -> dict:
    """
    Send chapter text to LLM for deep analysis.

    Args:
        chapter_content: Output from pdf_extractor.extract_pdf()
        output_path: Save analysis JSON here
        model: Override storyboard model

    Returns:
        Deep analysis dict
    """
    model = model or STORYBOARD_MODEL

    # Build context for the LLM
    full_text = chapter_content.get("full_text", "")
    # Truncate to fit context window (~30k chars should be safe for 128K context)
    if len(full_text) > 30000:
        full_text = full_text[:30000] + "\n\n[... truncated ...]"

    sections_summary = "\n".join(
        f"  {s['number']} — {s['title']}"
        for s in chapter_content.get("sections", [])
    )

    definitions_summary = "\n".join(
        f"  - {d[:200]}" for d in chapter_content.get("definitions", [])
    )

    examples_summary = "\n".join(
        f"  Example {e['number']}: {e['content'][:150]}..."
        for e in chapter_content.get("examples", [])
    )

    prompt = DEEP_ANALYSIS_PROMPT.format(
        chapter_text=full_text,
        sections_summary=sections_summary or "(none detected — use full text)",
        definitions_summary=definitions_summary or "(none detected — extract from text)",
        examples_summary=examples_summary or "(none detected — extract from text)",
    )

    print(f"[Analyzer] Sending {len(prompt)} chars to {model}...")
    print(f"[Analyzer] This may take 2-5 minutes for a thorough analysis...")

    client = ollama.Client(host=OLLAMA_BASE_URL)

    messages = [
        {"role": "system", "content": DEEP_ANALYSIS_SYSTEM},
        {"role": "user", "content": prompt},
    ]

    for attempt in range(3):
        response = client.chat(
            model=model,
            messages=messages,
            options={
                "temperature": STORYBOARD_TEMPERATURE,
                "num_predict": 16384,  # Allow long response
            },
            format="json",
        )

        text = response.get("message", {}).get("content", "").strip()

        try:
            analysis = _robust_json_parse(text)
            _validate_analysis(analysis)
            break
        except (json.JSONDecodeError, ValueError) as e:
            if attempt == 2:
                raise RuntimeError(f"Failed to generate valid deep analysis after 3 attempts: {e}")
            print(f"[Analyzer] Warning: LLM produced invalid JSON ({e}). Asking it to self-correct... (Attempt {attempt+1}/3)")
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"Your previous output was invalid JSON or failed validation: {e}. Please fix the formatting/content and return ONLY valid JSON."})

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"[Analyzer] Saved -> {output_path}")

    _print_summary(analysis)
    return analysis


def _validate_analysis(analysis: dict) -> None:
    """Validate the analysis has required fields."""
    required = ("core_concepts", "worked_examples", "chapter_flow")
    for key in required:
        if key not in analysis:
            raise ValueError(f"Analysis missing required field: '{key}'")

    if len(analysis.get("core_concepts", [])) < 3:
        print(f"[Analyzer] WARNING: Only {len(analysis.get('core_concepts', []))} core concepts found — expected more")

    if len(analysis.get("worked_examples", [])) < 3:
        print(f"[Analyzer] WARNING: Only {len(analysis.get('worked_examples', []))} examples found — expected more")


def _print_summary(analysis: dict) -> None:
    """Print a summary of the analysis."""
    print(f"\n{'='*60}")
    print(f"  Deep Analysis Complete")
    print(f"{'='*60}")
    print(f"  Chapter:        {analysis.get('chapter_title', 'Unknown')}")
    print(f"  Core concepts:  {len(analysis.get('core_concepts', []))}")
    print(f"  History events: {len(analysis.get('ancient_history', []))}")
    print(f"  Misconceptions: {len(analysis.get('misconceptions', []))}")
    print(f"  Fun facts:      {len(analysis.get('fun_facts', []))}")
    print(f"  Tricks:         {len(analysis.get('tricks_and_shortcuts', []))}")
    print(f"  Examples:       {len(analysis.get('worked_examples', []))}")
    print(f"  Chapter flow:   {len(analysis.get('chapter_flow', []))} steps")
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

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r'("\s*|\d+|true|false|null|\]|\})\s*\n\s*("|\{|\[)', r'\1,\n\2', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    fixed = re.sub(r',\s*([\}\]])', r'\1', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

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
        raise json.JSONDecodeError(f"Failed to parse LLM JSON (len {len(text)})", text, 0)
