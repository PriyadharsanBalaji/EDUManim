# Concept to Canvas Collaboration: Architectural Blueprint

Welcome to the **Concept to Canvas** repository. This is an advanced, fully automated educational pipeline. It autonomously ingests raw educational PDFs (like NCERT textbooks) and converts them into 30-60 minute, fully narrated, 3Blue1Brown-style Manim educational videos.

This README serves as the complete architectural blueprint of the system, detailing the exact internal workings, LLM prompts, and JSON schemas that power each stage.

---

## 🧠 The 6-Stage Pipeline Architecture

The entire process is orchestrated by `pipeline.py`. When you run the pipeline, it passes your PDF through 6 deeply integrated stages.

### Stage 0: PDF Text Extraction (`stages/pdf_extractor.py`)
This stage transforms the unstructured PDF into a machine-readable format.
- **Input:** Raw `.pdf` file.
- **Internal Mechanism:** Uses `PyMuPDF` to iterate through every page. It heuristically categorizes text blocks into regular paragraphs, section headings, and identifies structured elements like "Examples" and "Exercises".
- **Output:** `chapter_content.json`. This contains the raw `chapter_title`, `num_pages`, and a structured array of `sections` and `examples` extracted directly from the text.

### Stage 1: Deep Analysis (`stages/deep_analyzer.py`)
This stage acts as the "Subject Matter Expert". It doesn't write the script; it figures out *how* to teach the material.
- **Input:** `chapter_content.json`
- **The Prompt:** The LLM is instructed to act as an expert educational content analyst specializing in Indian school curricula. It is explicitly commanded to go *far beyond* the textbook by adding historical context, surprising connections, and exam tricks.
- **Output Schema (`deep_analysis.json`):** The LLM is forced to output this exact JSON structure:
  ```json
  {
      "ancient_history": [
          {"era": "...", "person": "...", "event": "...", "fun_angle": "..."}
      ],
      "core_concepts": [
          {"name": "...", "definition": "...", "intuition": "...", "visual_idea": "..."}
      ],
      "misconceptions": [
          {"wrong_belief": "...", "correct_understanding": "...", "visual_fix": "..."}
      ],
      "fun_facts": [],
      "tricks_and_shortcuts": [],
      "worked_examples": []
  }
  ```

### Stage 2: Storyboard Planning (`stages/storyboard_planner.py`)
This stage acts as the "Director". It takes the raw pedagogical concepts and maps them into a strict cinematic flow.
- **Input:** `deep_analysis.json`
- **The Prompt:** The LLM acts as an award-winning director creating 3Blue1Brown-style videos. We enforce a strict pedagogical arc: **Hook (1-2 scenes) → History (0-1 scenes) → Core Concepts (Intuition then Definition) → Misconceptions → Examples → Summary.**
- **Output Schema (`storyboard.json`):** This is the ultimate master blueprint. It generates an array of 50-80 individual "scenes". Each scene looks like this:
  ```json
  {
      "scene_id": 1,
      "teaching_beat": "HOOK",
      "narration_text": "The exact voiceover script to be read aloud (40-80 words).",
      "visual_description": "What the viewer physically sees on screen.",
      "manim_intent": "Precise Manim instructions (e.g., use FadeIn, Transform, VGroup).",
      "key_objects": ["MathTex", "Circle", "NumberLine"],
      "on_screen_text": ["Any specific LaTeX equations"]
  }
  ```

### Stage 3: Manim Code Generation (`stages/manim_codegen.py`)
This stage acts as the "Programmer". It translates the storyboard intent into executable Python code.
- **Input:** `storyboard.json`
- **The Process:** We loop through every single scene in the storyboard and feed it to the `manim-coder` model. 
- **The System Rules:** To prevent Manim hallucinations, the LLM is forced to obey 13 strict rules, including:
  1. Class names MUST be `Scene_XXX`.
  2. Inherit strictly from `Scene`.
  3. Manim coordinates MUST be 3D `[x, y, 0]` (prevents 2D array crashes).
  4. NEVER load external image files; use pure vector objects.
- **Output:** Raw `.py` script files dumped into the `scenes/` folder.

### Stage 4: Audio Generation (`stages/audio_generator.py`)
- **Input:** The `narration_text` strings from `storyboard.json`.
- **Process:** Connects to a Text-To-Speech (TTS) engine. It iterates through every scene and synthesizes a warm, engaging voiceover.
- **Output:** Voiceover `.mp3` files dumped into the `audio/` folder.

### Stage 5: Rendering & Assembly (`stages/renderer.py` & `stages/assembler.py`)
- **Input:** The `.py` Manim scripts and the `.mp3` audio files.
- **Process:** 
  1. `renderer.py` loops through all python files and runs `manim -qm` to compile them into `.mp4` video segments.
  2. `assembler.py` uses FFmpeg to multiplex each visual `.mp4` with its matching `.mp3`. If the audio is longer than the animation, the animation freezes on the last frame perfectly. 
  3. Finally, FFmpeg concatenates all 50-80 mixed scenes sequentially.
- **Output:** `final_video.mp4` — The fully completed educational video.

---

## 🧪 The LLM Comparison Workflow (`ComparisionPipeline/`)

How does our local `manim-coder` stack up against giant API models like **Claude** and **Gemini**? 

We built the `ComparisionPipeline/` directory as an evaluation sandbox to manually test this.

### The Evaluation Workflow:
1. **Extraction (`extract_manim_prompts.py`):** We "hijack" the `storyboard.json` generated at the end of Stage 2. This script converts the JSON back into exact `.txt` prompts containing the System Instructions (the 13 strict rules) and the User Prompt (the exact Scene intent).
2. **Generation:** You manually paste these prompts into the Claude/Gemini UI, and paste their generated python code into the `outputs/` folder.
3. **Automated Evaluation (`evaluators/render_and_compare.py`):** This script automatically tests the API LLM's code. It runs Manim locally, utilizing the `--media_dir` flag to isolate output videos so Claude and Gemini don't overwrite each other. It outputs a success/failure compile report.
4. **Stitching & Mixing (`evaluators/stitch_videos.py`):** This script automatically grabs the original TTS audio files from **Stage 4** of the main pipeline, frame-perfectly mixes them with Claude/Gemini's new animations using FFmpeg, and stitches them into a final comparative video.

---

## ⚙️ Execution Commands

**Run the Full Main Pipeline:**
```bash
python pipeline.py --pdf iemh101.pdf
```

**Generate the Storyboard Only (Stop after Stage 2):**
```bash
python pipeline.py --pdf iemh101.pdf --plan-only
```

**Resume from a specific stage (e.g., Stage 3: Code Gen):**
```bash
python pipeline.py --pdf iemh101.pdf --resume --stage 3
```

---

## 📝 Changelog

### V2
- **Reduced History Prompts:** The prompts in `deep_analyzer.py` and `storyboard_planner.py` have been heavily modified to reduce the amount of historical context generated. Previously, the pipeline asked for 4-6 scenes of history, which resulted in videos that were too heavily focused on history. In V2, the history requirement is reduced to 0-1 scenes for a more concise video.
