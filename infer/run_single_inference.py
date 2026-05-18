import argparse
import os
import sys
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer.unibiomed_inference_toolusing import (
    InferenceArgs,
    run_single_image_inference,
)
from infer.models.model_loader import load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-sample inference")
    parser.add_argument(
        "--img-path",
        type=str,
        default=str(PROJECT_ROOT / "infer" / "demo"/ "BTCV-0-106_CT_abdomen.png"),
        help="Path to input image",
    )
    parser.add_argument(
        "--target-description",
        type=str,
        default="right kidney in abdomen CT",
        help="Target description text",
    )
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
    parser.add_argument("--n-clicks", type=int, default=5, help="Max clicks")
    parser.add_argument(
        "--grounding-model",
        type=str,
        default="gemma",
        choices=["gemma"],
        help="Grounding model type",
    )
    parser.add_argument(
        "--seg-config",
        type=str,
        default=None,
        help="MedSAM2 config path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "infer" / "intermediate_result"),
        help="Directory to save inference records and visualizations",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(PROJECT_ROOT / "infer" / "results"),
        help="Directory to save final results",
    )
    parser.add_argument(
        "--grounding-resize",
        type=int,
        default=512,
        help="Resize resolution for grounding model input (set 0 to disable)",
    )
    parser.add_argument(
        "--max-history-length",
        type=int,
        default=5,
        help="Max history length (only for history-enabled models)",
    )
    return parser.parse_args()


def main():
    cli = parse_args()

    if not os.path.exists(cli.img_path):
        raise FileNotFoundError(f"Image not found: {cli.img_path}")

    args = InferenceArgs()
    args.n_clicks = cli.n_clicks
    args.grounding_model = cli.grounding_model
    args.seg_model = "medsam"
    args.output_dir = cli.output_dir
    args.results_dir = cli.results_dir
    args.grounding_resize = None if cli.grounding_resize == 0 else cli.grounding_resize

    # History settings
    args.max_history_length = cli.max_history_length
    args.reset_history_per_image = True

    # Segmentation settings
    if cli.seg_checkpoint:
        args.seg_checkpoint = cli.seg_checkpoint
    if cli.seg_config:
        args.seg_config = cli.seg_config

    if not args.seg_config:
        args.seg_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
    # Gemma settings
    args.model = cli.model_path

    # Dataset name is used by Clicker for saving
    args.dataset_name = "demo"

    print("Loading models...")
    segmentation_model, grounding_model = load_model(args)

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
