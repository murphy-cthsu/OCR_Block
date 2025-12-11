#!/usr/bin/env python3
"""
Batch annotation script for block diagrams.

Processes all images in the images/ directory and saves results to outputs/
with the format: {filename}_final_annotation.json and {filename}_intermediate_*.json
"""

import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

from annotate import BlockDiagramAnnotator


# Supported image extensions
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}


def get_image_files(images_dir: Path) -> list[Path]:
    """Get all supported image files from the images directory."""
    image_files = []
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(images_dir.glob(f"*{ext}"))
        image_files.extend(images_dir.glob(f"*{ext.upper()}"))
    return sorted(set(image_files))


def process_single_image(
    annotator: BlockDiagramAnnotator,
    image_path: Path,
    output_dir: Path,
    multi_pass: bool = True,
    multi_model: bool = False,
    save_intermediate: bool = True,
    skip_existing: bool = False
) -> dict | None:
    """
    Process a single image and save results to output directory.
    
    Args:
        annotator: BlockDiagramAnnotator instance
        image_path: Path to the image file
        output_dir: Directory to save output files
        multi_pass: Whether to do multiple passes
        multi_model: Whether to use multiple models
        save_intermediate: Whether to save intermediate results
        skip_existing: Skip if final output already exists
    
    Returns:
        Annotation result dict or None if skipped/failed
    """
    filename_stem = image_path.stem  # e.g., "MT6797" from "MT6797.png"
    final_output_path = output_dir / f"{filename_stem}_final_annotation.json"
    
    # Skip if already processed
    if skip_existing and final_output_path.exists():
        print(f"\n{'='*60}")
        print(f"SKIPPING: {image_path.name} (already processed)")
        print(f"{'='*60}")
        return None
    
    print(f"\n{'#'*60}")
    print(f"PROCESSING: {image_path.name}")
    print(f"{'#'*60}")
    
    try:
        # Run annotation with output directory and prefix
        result = annotator.annotate(
            str(image_path),
            multi_pass=multi_pass,
            multi_model=multi_model,
            save_intermediate=save_intermediate,
            output_dir=str(output_dir),
            output_prefix=filename_stem
        )
        
        # Save final result
        with open(final_output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved final result to: {final_output_path}")
        
        return result
        
    except Exception as e:
        # Re-raise fatal API errors
        if "Key limit exceeded" in str(e) or "Insufficient credits" in str(e):
            raise e
            
        print(f"\nERROR processing {image_path.name}: {e}")
        # Save error log
        error_log_path = output_dir / f"{filename_stem}_error.log"
        with open(error_log_path, "w", encoding="utf-8") as f:
            f.write(f"Error processing {image_path.name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Error: {str(e)}\n")
        return None


def batch_annotate(
    images_dir: str | Path = "images",
    output_dir: str | Path = "outputs",
    multi_pass: bool = True,
    multi_model: bool = False,
    save_intermediate: bool = True,
    skip_existing: bool = True,
    specific_files: list[str] | None = None
) -> dict:
    """
    Batch process all images in the images directory.
    
    Args:
        images_dir: Directory containing block diagram images
        output_dir: Directory to save output JSON files
        multi_pass: Whether to do multiple passes with primary model
        multi_model: Whether to use multiple models for cross-validation
        save_intermediate: Whether to save intermediate pass results
        skip_existing: Skip images that already have final output
        specific_files: Optional list of specific filenames to process
    
    Returns:
        Summary dict with processing results
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image files
    if specific_files:
        image_files = [images_dir / f for f in specific_files if (images_dir / f).exists()]
    else:
        image_files = get_image_files(images_dir)
    
    if not image_files:
        print(f"No image files found in {images_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}
    
    print(f"\n{'='*60}")
    print(f"BATCH ANNOTATION")
    print(f"{'='*60}")
    print(f"Images directory: {images_dir.absolute()}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Total images found: {len(image_files)}")
    print(f"Multi-pass: {multi_pass}")
    print(f"Multi-model: {multi_model}")
    print(f"Save intermediate: {save_intermediate}")
    print(f"Skip existing: {skip_existing}")
    print(f"{'='*60}")
    
    # Initialize annotator
    annotator = BlockDiagramAnnotator()
    
    # Process results tracking
    results = {
        "processed": [],
        "skipped": [],
        "failed": []
    }
    
    start_time = datetime.now()
    
    for idx, image_path in enumerate(image_files, 1):
        print(f"\n[{idx}/{len(image_files)}] ", end="")
        
        final_output = output_dir / f"{image_path.stem}_final_annotation.json"
        
        if skip_existing and final_output.exists():
            print(f"Skipping {image_path.name} (already exists)")
            results["skipped"].append(image_path.name)
            continue
        
        try:
            result = process_single_image(
                annotator=annotator,
                image_path=image_path,
                output_dir=output_dir,
                multi_pass=multi_pass,
                multi_model=multi_model,
                save_intermediate=save_intermediate,
                skip_existing=skip_existing
            )
            
            if result is not None:
                results["processed"].append(image_path.name)
            elif not (skip_existing and final_output.exists()):
                results["failed"].append(image_path.name)
                
        except Exception as e:
            print(f"\n\nCRITICAL ERROR: {e}")
            print("Stopping batch processing immediately.")
            results["failed"].append(image_path.name)
            break
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print("BATCH ANNOTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Duration: {duration}")
    print(f"Processed: {len(results['processed'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"Failed: {len(results['failed'])}")
    
    if results["failed"]:
        print(f"\nFailed files:")
        for f in results["failed"]:
            print(f"  - {f}")
    
    # Save batch summary
    summary_path = output_dir / "batch_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration.total_seconds(),
        "settings": {
            "multi_pass": multi_pass,
            "multi_model": multi_model,
            "save_intermediate": save_intermediate
        },
        "results": {
            "processed": results["processed"],
            "skipped": results["skipped"],
            "failed": results["failed"]
        }
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nBatch summary saved to: {summary_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch annotate block diagram images"
    )
    parser.add_argument(
        "--images-dir", "-i",
        default="images",
        help="Directory containing block diagram images (default: images)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="outputs",
        help="Directory to save output JSON files (default: outputs)"
    )
    parser.add_argument(
        "--no-multi-pass",
        action="store_true",
        help="Disable multiple passes with primary model"
    )
    parser.add_argument(
        "--multi-model",
        action="store_true",
        help="Enable multiple models for cross-validation"
    )
    parser.add_argument(
        "--no-intermediate",
        action="store_true",
        help="Don't save intermediate pass results"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-process images even if output already exists"
    )
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        help="Specific image files to process (e.g., MT6797.png MT8395.png)"
    )
    
    args = parser.parse_args()
    
    batch_annotate(
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        multi_pass=not args.no_multi_pass,
        multi_model=args.multi_model,
        save_intermediate=not args.no_intermediate,
        skip_existing=not args.force,
        specific_files=args.files
    )


if __name__ == "__main__":
    main()
