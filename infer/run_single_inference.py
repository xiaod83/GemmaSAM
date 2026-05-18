"""
CLI entry point for GemmaSAM single-image inference.

Parses command-line arguments, loads the grounding model (Gemma 4) and
segmentation model (MedSAM2), then runs the interactive VLM-guided
segmentation loop on a single medical image.
"""

import argparse
import os
import sys
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
# Ensure the project root is on sys.path so all infer.* imports resolve
# regardless of which directory the script is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer.unibiomed_inference_toolusing import (
    InferenceArgs,
    run_single_image_inference,
)
from infer.models.model_loader import load_model


def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments for a single inference run."""
    parser = argparse.ArgumentParser(description="Run single-sample inference")

    # ── Input data ──────────────────────────────────────────────────────
    parser.add_argument(
        "--img-path",
        type=str,
        default=str(PROJECT_ROOT / "infer" / "demo"/ "Normal_abdominal_organs_CT_scan.png"),
        help="Path to input image",
    )
    parser.add_argument(
        "--target-description",
        type=str,
        default="liver",
        help="Target description text (e.g. 'liver', 'brain tumor')",
    )

    # ── Model paths ─────────────────────────────────────────────────────
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local Gemma checkpoint dir (used when grounding-model=gemma)",
    )
    parser.add_argument(
        "--seg-checkpoint",
        type=str,
        required=True,
        help="MedSAM2 checkpoint path",
    )

    # ── Segmentation settings ───────────────────────────────────────────
    parser.add_argument("--n-clicks", type=int, default=5, help="Maximum number of click/bbox rounds")
    parser.add_argument(
        "--grounding-model",
        type=str,
        default="gemma",
        choices=["gemma"],
        help="Grounding model type (only gemma is currently supported)",
    )
    parser.add_argument(
        "--seg-config",
        type=str,
        default=None,
        help="MedSAM2 config path (defaults to sam2.1_hiera_t.yaml)",
    )

    # ── Output directories ──────────────────────────────────────────────
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "infer" / "intermediate_result"),
        help="Directory to save per-image inference records and visualizations",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "infer" / "results"),
        help="Directory to save aggregated batch results",
    )

    # ── Model / inference tuning ────────────────────────────────────────
    parser.add_argument(
        "--grounding-resize",
        type=int,
        default=512,
        help="Resize resolution for grounding model input (set 0 to disable resize)",
    )
    parser.add_argument(
        "--max-history-length",
        type=int,
        default=5,
        help="Max conversation-history turns (only for history-enabled models like Gemma)",
    )

    return parser.parse_args()


def main() -> None:
    """Load models, configure inference, and run the segmentation loop."""
    cli = parse_args()

    # Validate inputs early
    if not os.path.exists(cli.img_path):
        raise FileNotFoundError(f"Image not found: {cli.img_path}")

    # ── Build inference args ────────────────────────────────────────────
    args = InferenceArgs()
    args.n_clicks = cli.n_clicks
    args.grounding_model = cli.grounding_model
    args.seg_model = "medsam"                              # MedSAM2 is the only supported segmenter
    args.output_dir = cli.output_dir
    args.results_dir = cli.results_dir
    args.grounding_resize = None if cli.grounding_resize == 0 else cli.grounding_resize

    # Conversation history — one session per image, reset between images
    args.max_history_length = cli.max_history_length
    args.reset_history_per_image = True

    # Segmentation checkpoint / config
    if cli.seg_checkpoint:
        args.seg_checkpoint = cli.seg_checkpoint
    if cli.seg_config:
        args.seg_config = cli.seg_config
    if not args.seg_config:
        # Default MedSAM2 tiny config bundled with sam2 repo
        args.seg_config = "configs/sam2.1/sam2.1_hiera_t.yaml"

    # Gemma 4 checkpoint path
    args.model = cli.model_path

    # Used by Clicker when naming saved artifacts
    args.dataset_name = "demo"

    # ── Load models ─────────────────────────────────────────────────────
    print("Loading models...")
    segmentation_model, grounding_model = load_model(args)

    # ── Run interactive segmentation ────────────────────────────────────
    print("Running single-image inference...")
    final_mask, record_path = run_single_image_inference(
        img_path=cli.img_path,
        target_description=cli.target_description,
        grounding_model=grounding_model,
        segmentation_model=segmentation_model,
        args=args,
        max_clicks=args.n_clicks,
    )

    if final_mask is None:
        raise RuntimeError("Inference failed: no valid mask returned")

    print(f"Done. Inference record saved to: {record_path}")


if __name__ == "__main__":
    main()
