"""
pipeline.py — Main orchestrator for the Manim Educational Video Pipeline.

Takes a PDF chapter → Deep Analysis → Storyboard → Manim Code → Render → Assemble.
All stages are resumable. Run with --plan-only to stop after storyboard generation.

Usage:
    python pipeline.py --pdf iemh101.pdf
    python pipeline.py --pdf iemh101.pdf --plan-only
    python pipeline.py --pdf iemh101.pdf --resume --stage 3
"""

import json
import time
from pathlib import Path

from config import DEFAULT_OUTPUT_DIR, MANIM_QUALITY


def run_pipeline(
    pdf_path: str,
    output_name: str = None,
    target_minutes: int = 45,
    resume: bool = True,
    plan_only: bool = False,
    start_stage: int = 0,
    quality: str = None,
    storyboard_model: str = None,
    manim_model: str = None,
    pre_extracted_content: dict = None,
) -> dict:
    """
    End-to-end pipeline: PDF → educational Manim video (30-60 min).

    Args:
        pdf_path:         Path to the input PDF file
        output_name:      Slug for output directory (auto-derived from PDF name if None)
        target_minutes:   Target video duration in minutes (30-60)
        resume:           Resume from existing state if interrupted
        plan_only:        Stop after Stage 2 (storyboard generation)
        start_stage:      Skip to this stage (0-5)
        quality:          Manim render quality (l/m/h/k)
        storyboard_model: Override storyboard LLM model name
        manim_model:      Override manim-coder model name
        pre_extracted_content: Extracted PDF content to bypass PDF reading (useful for chunking)

    Returns:
        Dict with pipeline results
    """
    quality = quality or MANIM_QUALITY
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not output_name:
        output_name = pdf_path.stem.lower().replace(" ", "_")

    # ── Directory setup ──────────────────────────────────────────────────────
    base_dir = Path(DEFAULT_OUTPUT_DIR) / output_name
    scenes_dir = base_dir / "scenes"
    renders_dir = base_dir / "renders"
    audio_dir = base_dir / "audio"

    base_dir.mkdir(parents=True, exist_ok=True)
    scenes_dir.mkdir(parents=True, exist_ok=True)
    renders_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    # State files
    chapter_content_file = str(base_dir / "chapter_content.json")
    deep_analysis_file = str(base_dir / "deep_analysis.json")
    storyboard_file = str(base_dir / "storyboard.json")
    codegen_state_file = str(base_dir / "codegen_state.json")
    render_state_file = str(base_dir / "render_state.json")
    audio_state_file = str(base_dir / "audio_state.json")
    final_output = str(base_dir / f"final_{output_name}.mp4")

    start_time = time.time()

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 0 — PDF Text Extraction
    # ════════════════════════════════════════════════════════════════════════
    if start_stage <= 0:
        _header("STAGE 0 — PDF Text Extraction (PyMuPDF)")

        if resume and Path(chapter_content_file).exists():
            print(f"Loading existing chapter content from {chapter_content_file}")
            with open(chapter_content_file, encoding="utf-8") as f:
                chapter_content = json.load(f)
        elif pre_extracted_content:
            print("Using pre-extracted chapter content for chunk processing.")
            chapter_content = pre_extracted_content
            # Save it so resume logic works for subsequent stages
            with open(chapter_content_file, "w", encoding="utf-8") as f:
                json.dump(chapter_content, f, indent=2, ensure_ascii=False)
        else:
            from stages.pdf_extractor import extract_pdf
            chapter_content = extract_pdf(str(pdf_path), output_path=chapter_content_file)

        print(f"✓ Chapter: {chapter_content.get('chapter_title', 'Unknown')}")
        print(f"  Pages: {chapter_content.get('num_pages', '?')}, "
              f"Sections: {len(chapter_content.get('sections', []))}, "
              f"Examples: {len(chapter_content.get('examples', []))}")
    else:
        with open(chapter_content_file, encoding="utf-8") as f:
            chapter_content = json.load(f)

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 1 — Deep Concept Analysis (Ollama)
    # ════════════════════════════════════════════════════════════════════════
    if start_stage <= 1:
        _header("STAGE 1 — Deep Concept Analysis (Ollama)")

        if resume and Path(deep_analysis_file).exists():
            print(f"Loading existing analysis from {deep_analysis_file}")
            with open(deep_analysis_file, encoding="utf-8") as f:
                deep_analysis = json.load(f)
        else:
            from stages.deep_analyzer import analyze_chapter
            deep_analysis = analyze_chapter(
                chapter_content,
                output_path=deep_analysis_file,
                model=storyboard_model,
            )

        print(f"✓ Analysis complete — {len(deep_analysis.get('core_concepts', []))} concepts, "
              f"{len(deep_analysis.get('worked_examples', []))} examples")
    else:
        with open(deep_analysis_file, encoding="utf-8") as f:
            deep_analysis = json.load(f)

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 2 — Storyboard Generation (Ollama)
    # ════════════════════════════════════════════════════════════════════════
    if start_stage <= 2:
        _header("STAGE 2 — Storyboard Generation (Ollama)")

        if resume and Path(storyboard_file).exists():
            print(f"Loading existing storyboard from {storyboard_file}")
            with open(storyboard_file, encoding="utf-8") as f:
                storyboard = json.load(f)
        else:
            from stages.storyboard_planner import plan_storyboard
            storyboard = plan_storyboard(
                deep_analysis,
                output_path=storyboard_file,
                model=storyboard_model,
                target_minutes=target_minutes,
            )

        total_scenes = storyboard.get("total_scenes", "?")
        print(f"✓ Storyboard: {total_scenes} scenes across "
              f"{len(storyboard.get('sections', []))} sections")
        print(f"  Review: {base_dir / 'STORYBOARD.md'}")
    else:
        with open(storyboard_file, encoding="utf-8") as f:
            storyboard = json.load(f)

    if plan_only:
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"⚡ PLAN ONLY — review STORYBOARD.md, then re-run without --plan-only")
        print(f"  Completed in {elapsed:.1f}s")
        print(f"{'='*60}")
        return {
            "chapter_content": chapter_content_file,
            "deep_analysis": deep_analysis_file,
            "storyboard": storyboard_file,
            "storyboard_md": str(base_dir / "STORYBOARD.md"),
        }

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 3 — Manim Code Generation (Ollama manim-coder)
    # ════════════════════════════════════════════════════════════════════════
    if start_stage <= 3:
        _header("STAGE 3 — Manim Code Generation (manim-coder)")
        print("⏱  Each scene takes ~10-30 seconds to generate code.")

        from stages.manim_codegen import generate_all_scenes
        scene_files = generate_all_scenes(
            storyboard=storyboard,
            output_dir=str(scenes_dir),
            state_file=codegen_state_file,
            model=manim_model,
        )

        print(f"✓ {len(scene_files)} scene scripts generated in {scenes_dir}")
    else:
        # Reconstruct scene_files from codegen state
        with open(codegen_state_file, encoding="utf-8") as f:
            codegen_state = json.load(f)
        scene_files = {
            sid: info["file"]
            for sid, info in codegen_state.items()
            if info.get("status") == "done"
        }

    if not scene_files:
        raise RuntimeError("No scene files generated. Check Ollama connection and manim-coder model.")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 4 — Manim Rendering
    # ════════════════════════════════════════════════════════════════════════
    if start_stage <= 4:
        _header("STAGE 4 — Manim Rendering")
        print(f"⏱  Rendering {len(scene_files)} scenes at quality={quality}.")
        print(f"   This may take 5-30 minutes depending on scene complexity.")

        from stages.renderer import render_all_scenes
        from stages.manim_codegen import fix_scene_code

        # Self-healing callback: if render fails, ask manim-coder to fix code using traceback
        def _fix_scene(scene_id, py_path, error):
            print(f"    [Self-heal] Sending render error traceback to manim-coder...")
            # Find the scene data from storyboard
            for section in storyboard.get("sections", []):
                for scene in section.get("scenes", []):
                    if scene.get("global_scene_id") == scene_id:
                        fixed = fix_scene_code(
                            scene=scene,
                            scene_id=scene_id,
                            output_path=py_path,
                            render_error=error,
                            model=manim_model,
                        )
                        return py_path if fixed else None
            return None

        rendered_files = render_all_scenes(
            scene_files=scene_files,
            output_dir=str(renders_dir),
            state_file=render_state_file,
            quality=quality,
            manim_codegen_fix_fn=_fix_scene,
        )

        print(f"✓ {len(rendered_files)} scenes rendered")
    else:
        with open(render_state_file, encoding="utf-8") as f:
            render_state = json.load(f)
        rendered_files = {
            sid: info["file"]
            for sid, info in render_state.items()
            if info.get("status") == "rendered"
        }

    if not rendered_files:
        raise RuntimeError("No scenes rendered. Check Manim installation and LaTeX availability.")

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 4.5 — Voiceover Audio Generation (edge-tts / gTTS)
    # ════════════════════════════════════════════════════════════════════════
    _header("STAGE 4.5 — Voiceover Audio Generation")
    from stages.audio_generator import generate_scene_audio
    audio_files = generate_scene_audio(
        storyboard=storyboard,
        output_dir=str(audio_dir),
        state_file=audio_state_file,
    )

    # ════════════════════════════════════════════════════════════════════════
    # STAGE 5 — Video Assembly & Audio Merging (FFmpeg)
    # ════════════════════════════════════════════════════════════════════════
    _header("STAGE 5 — Video Assembly & Audio Merging (FFmpeg)")

    from stages.assembler import assemble_video, merge_scene_audio_video
    
    combined_files = merge_scene_audio_video(
        rendered_files=rendered_files,
        audio_files=audio_files,
        output_dir=str(base_dir / "clips_with_audio"),
    )

    final_path = assemble_video(
        rendered_files=combined_files,
        storyboard=storyboard,
        output_path=final_output,
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ PIPELINE COMPLETE in {elapsed / 60:.1f} minutes")
    print(f"{'='*60}")
    print(f"  Output    : {final_path}")
    print(f"  Scenes    : {len(rendered_files)} rendered")
    print(f"  Title     : {storyboard.get('title', 'Untitled')}")
    print(f"  Storyboard: {base_dir / 'STORYBOARD.md'}")
    print(f"{'='*60}\n")

    return {
        "output": final_path,
        "storyboard": storyboard_file,
        "total_scenes": len(rendered_files),
        "elapsed_minutes": elapsed / 60,
    }


def _header(title: str) -> None:
    print(f"\n{'═'*60}")
    print(f"  {title}")
    print(f"{'═'*60}")
