import os
import sys
import time
import subprocess
import traceback
from pathlib import Path
import copy

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import run_pipeline
from config import STORYBOARD_MODEL, MANIM_MODEL
from stages.pdf_extractor import extract_pdf

def pull_models():
    """Ensure required Ollama models are pulled."""
    print("============================================================")
    print(" Pulling required Ollama models...")
    print("============================================================")
    models = [STORYBOARD_MODEL, MANIM_MODEL]
    for model in models:
        try:
            print(f"Pulling {model}...")
            # We don't want to block forever if ollama server isn't running, but subprocess should handle it.
            result = subprocess.run(["ollama", "pull", model], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"Successfully pulled (or already have) {model}")
            else:
                print(f"Warning: Failed to pull {model}. Output:\n{result.stderr}")
        except Exception as e:
            print(f"Warning: Could not run ollama pull for {model}. Is Ollama installed and running? Error: {e}")

import argparse

def main():
    parser = argparse.ArgumentParser(description="Batch run the Manim pipeline.")
    parser.add_argument("--first", type=str, help="Name of the PDF file to process first (e.g., 'Integers.pdf')")
    parser.add_argument("--whole-book", action="store_true", help="Process the entire book iteratively by chunking")
    parser.add_argument("--iterations-per-book", type=int, default=None, help="Max iterations/chunks to process per book")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  Manim Educational Video Pipeline — Batch Runner")
    print("="*60 + "\n")

    # 1. Pull models
    pull_models()

    # 2. Find PDFs
    inputs_dir = Path("inputs")
    if not inputs_dir.exists():
        print(f"Error: Inputs directory '{inputs_dir.absolute()}' does not exist.")
        sys.exit(1)

    pdf_files = list(inputs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {inputs_dir.absolute()}")
        sys.exit(0)

    # Prioritize the specified book if --first is provided
    if args.first:
        first_pdf = next((p for p in pdf_files if p.name.lower() == args.first.lower()), None)
        if first_pdf:
            pdf_files.remove(first_pdf)
            pdf_files.insert(0, first_pdf)
            print(f"Prioritizing {first_pdf.name} to run first.")
        else:
            print(f"Warning: Could not find '{args.first}' in inputs directory.")

    print(f"\nFound {len(pdf_files)} PDF(s) to process:\n" + "\n".join([f"  - {p.name}" for p in pdf_files]))

    # 3. Process each PDF
    successes = []
    failures = []

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n\n{'#'*60}")
        print(f" Processing {i}/{len(pdf_files)}: {pdf_path.name}")
        print(f"{'#'*60}\n")
        
        start_time = time.time()
        
        try:
            output_name = pdf_path.stem.lower().replace(" ", "_")
            
            if args.whole_book:
                print(f"--- Extracting entire book for chunking ---")
                full_content = extract_pdf(str(pdf_path))
                sections = full_content.get("sections", [])
                
                # Chunking strategy: 2 sections per chunk
                chunk_size = 2
                chunks = [sections[i:i + chunk_size] for i in range(0, len(sections), chunk_size)]
                
                max_iter = args.iterations_per_book if args.iterations_per_book else len(chunks)
                chunk_videos = []
                
                for idx in range(min(max_iter, len(chunks))):
                    print(f"\n>> Processing Chunk {idx+1}/{len(chunks)}")
                    chunk_content = copy.deepcopy(full_content)
                    chunk_content["sections"] = chunks[idx]
                    
                    chunk_output_name = f"{output_name}_part{idx+1}"
                    result = run_pipeline(
                        pdf_path=str(pdf_path),
                        output_name=chunk_output_name,
                        target_minutes=15, # smaller target for chunk
                        resume=True,
                        plan_only=False,
                        start_stage=0,
                        quality="m",
                        pre_extracted_content=chunk_content
                    )
                    if result and "output" in result and os.path.exists(result["output"]):
                        chunk_videos.append(result["output"])
                
                # Stitch videos
                if chunk_videos:
                    print(f"\n--- Stitching {len(chunk_videos)} chunk videos together ---")
                    list_file_path = f"inputs/{output_name}_vid_list.txt"
                    with open(list_file_path, "w") as lf:
                        for vid in chunk_videos:
                            # ffmpeg needs absolute paths or relative paths in proper format
                            # Using forward slashes for ffmpeg compatibility
                            vid_path = str(Path(vid).absolute()).replace('\\', '/')
                            lf.write(f"file '{vid_path}'\n")
                    
                    final_merged_vid = f"outputs/{output_name}_complete.mp4"
                    stitch_cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file_path, "-c", "copy", final_merged_vid]
                    subprocess.run(stitch_cmd, check=True)
                    print(f"✅ Final merged video available at: {final_merged_vid}")
            
            else:
                # Original logic
                result = run_pipeline(
                    pdf_path=str(pdf_path),
                    output_name=output_name,
                    target_minutes=45,       # default target
                    resume=True,             # resume if interrupted
                    plan_only=False,         # full run
                    start_stage=0,           # from beginning
                    quality="m",             # medium quality 720p
                )
            
            elapsed = time.time() - start_time
            print(f"\n✅ Successfully processed {pdf_path.name} in {elapsed/60:.1f} minutes")
            successes.append(pdf_path.name)
            
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"\n❌ FAILED to process {pdf_path.name} after {elapsed/60:.1f} minutes")
            print("Error details:")
            traceback.print_exc()
            failures.append((pdf_path.name, str(e)))
            print("\nContinuing to the next PDF...")

    # 4. Summary
    print("\n" + "="*60)
    print(" BATCH RUN SUMMARY")
    print("="*60)
    print(f"Total processed: {len(pdf_files)}")
    print(f"Successes: {len(successes)}")
    print(f"Failures: {len(failures)}")
    
    if failures:
        print("\nFailed files:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    
    print("\nBatch run completed.")

if __name__ == "__main__":
    main()
