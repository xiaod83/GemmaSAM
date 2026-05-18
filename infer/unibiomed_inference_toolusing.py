"""
Core inference engine for GemmaSAM.

Contains the interactive segmentation loop, the Gemma 4 grounding model
wrapper (with conversation history support), evaluation metrics, and the
convenience orchestration function.

Architecture overview
---------------------
  User text prompt ──► GroundingModel_Gemma_WithHistory ──► tool_call JSON
                              │                                    │
                              │  (overlaid mask image)             │ add_bbox / add_point
                              ◄────────────────────────────────────◄
                              │
                              ▼
                      SingleImageInference.forward_single_image()
                              │
                              ▼
                      MedSAM2 (segmentation)
                              │
                              ▼
                      Final mask + inference_record.json
"""

import json
import os
import cv2
import numpy as np
import torch

from tqdm import tqdm
from PIL import Image

from utils.visual_utils import (
    visualize_mask_and_pointlist,
    overlay_mask,
)
from utils.clicker import Click, Clicker


# ═══════════════════════════════════════════════════════════════════════════
#  Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_metrics(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Compute Dice coefficient and IoU between a predicted mask and ground truth.

    Args:
        prediction: Binary predicted mask.
        target: Binary ground-truth mask.

    Returns:
        (dice, iou) tuple, each in [0, 1].
    """
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()

    dice = 2.0 * intersection / (prediction.sum() + target.sum() + 1e-8)
    IoU = intersection / (union + 1e-8)

    return dice, IoU


def get_mean(save_metrics: dict) -> dict:
    """Compute per-class and overall mean Dice / IoU from a dict of per-sample metrics.

    Args:
        save_metrics: Dict keyed by mask filename, each containing 'dice', 'IoU', 'exp' (class name).

    Returns:
        Dict with per-class + overall mean values.
    """
    mean_metrics: dict = {}
    classes_list: list[str] = []

    # Collect unique class names
    for name in save_metrics.keys():
        each_item = save_metrics[name]
        if 'exp' in each_item.keys():
            class_name = each_item['exp']
            if class_name not in classes_list:
                classes_list.append(class_name)

    all_class_dice: list[float] = []
    all_class_iou: list[float] = []

    all_dice: list[float] = []
    all_iou: list[float] = []

    for per_class_name in classes_list:
        per_class_metrics: dict = {}

        dice_per_class_list: list[float] = []
        iou_per_class_list: list[float] = []

        for name in save_metrics.keys():
            each_item = save_metrics[name]
            if 'exp' in each_item.keys():
                class_name = each_item['exp']
                if class_name == per_class_name:
                    dice_per_class_list.append(each_item['dice'])
                    iou_per_class_list.append(each_item['IoU'])
                all_dice.append(each_item['dice'])
                all_iou.append(each_item['IoU'])

        # Mean for this class (tiny epsilon avoids division by zero on empty lists)
        dice_per_class = sum(dice_per_class_list) / (len(dice_per_class_list) + 1e-6)
        iou_per_class = sum(iou_per_class_list) / (len(iou_per_class_list) + 1e-6)

        print(per_class_name, dice_per_class)
        per_class_metrics['dice'] = dice_per_class
        per_class_metrics['IoU'] = iou_per_class

        all_class_dice.append(dice_per_class)
        all_class_iou.append(iou_per_class)

        mean_metrics[per_class_name] = per_class_metrics

    # Grand mean of class-wise means (treats each class equally)
    mean_class_dice = sum(all_class_dice) / (len(all_class_dice) + 1e-6)
    mean_class_iou = sum(all_class_iou) / (len(all_class_iou) + 1e-6)
    mean_metrics['mean_class_dice'] = mean_class_dice
    mean_metrics['mean_class_iou'] = mean_class_iou

    # Overall mean across all samples (treats each sample equally)
    mean_dice = sum(all_dice) / (len(all_dice) + 1e-6)
    mean_iou = sum(all_iou) / (len(all_iou) + 1e-6)
    mean_metrics['mean_dice'] = mean_dice
    mean_metrics['mean_iou'] = mean_iou

    print('mean_class_dice, mean_class_iou:', mean_class_dice, mean_class_iou)
    print('mean_dice, mean_iou:', mean_dice, mean_iou)

    return mean_metrics


def save_metrics_json(data_root: str, results_root: str, val_json_file: str = 'train.json') -> None:
    """Batch evaluation: compare predicted masks against ground truth and save metrics.

    Reads a JSON annotation file pointing to ground-truth masks, finds corresponding
    predictions under results_root, computes Dice/IoU per sample, averages by class,
    and writes everything to results_root/results.json.

    Args:
        data_root: Base directory containing the annotation JSON and mask folders.
        results_root: Directory with predicted mask files (subfolders per split).
        val_json_file: Filename of the annotation JSON under data_root.
    """
    save_json_file = os.path.join(results_root, 'results.json')

    json_file = os.path.join(data_root, val_json_file)
    with open(json_file, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']

    save_metrics: dict = {}

    for item in annotations:
        item_metric: dict = {}

        mask_name = item['mask_file']

        # Ground-truth mask path
        mask = os.path.join(data_root, item['split'] + '_mask', mask_name)
        # Predicted mask path
        pred = os.path.join(results_root, item['split'], mask_name)

        mask_im, pred_im = Image.open(mask), Image.open(pred)
        mask_arr, pred_arr = np.asarray(mask_im), np.asarray(pred_im)

        # If GT has multiple channels, use only the first
        if len(mask_arr.shape) == 3:
            mask_arr = mask_arr[:, :, 0]

        # Resize GT to match prediction if shapes differ
        if mask_arr.shape != pred_arr.shape:
            mask_arr = cv2.resize(mask_arr, (pred_arr.shape[1], pred_arr.shape[0]))

        # Binarize both
        mask_arr = (mask_arr > 0).astype(np.uint8)
        pred_arr = (pred_arr > 0).astype(np.uint8)

        # Only compute metrics when GT has at least some foreground
        if mask_arr.sum() > 0:
            dice, IoU = get_metrics(pred_arr, mask_arr)
            print('mask_name, dice, IoU:', mask_name, dice, IoU)

            item_metric['dice'] = dice
            item_metric['IoU'] = IoU
            # Use the first sentence from the annotation as the class/expression label
            item_metric['exp'] = item['sentences'][0]['sent']

        save_metrics[mask_name] = item_metric
        print(item_metric)
        print('\n')

    mean_metrics = get_mean(save_metrics)

    # Write mean metrics first for easy parsing
    new_save_metrics: dict = {}
    new_save_metrics['mean'] = mean_metrics

    for key in save_metrics.keys():
        new_save_metrics[key] = save_metrics[key]

    with open(save_json_file, 'w', encoding='utf-8') as json_file:
        json.dump(new_save_metrics, json_file, ensure_ascii=False, indent=4)


# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

class InferenceArgs:
    """Configuration container for a single inference run.

    All inference-relevant knobs live here so they can be set from the CLI
    entry point (run_single_inference.py) or programmatically.
    """

    def __init__(self):
        # ── Basic parameters ────────────────────────────────────────────
        self.n_clicks = 5                    # Maximum VLM click/bbox rounds
        self.visualize = True                # Save per-step visualization images

        # Resize the image to this resolution before feeding it to the
        # grounding VLM.  The segmentation model always sees the full
        # original resolution.  Set to None to disable.
        self.grounding_resize = 512

        # Input resolution expected internally by the segmentation model.
        # (The model_loader uses this for initialization; inference itself
        #  runs on the original image dimensions.)
        self.seg_image_size = 1024

        # ── Performance ─────────────────────────────────────────────────
        self.use_fp16 = False                # BF16/FP16 for the VLM (reduces VRAM)
        self.batch_size = 1                  # Batch size (reserved for future use)

        # ── Segmentation ────────────────────────────────────────────────
        self.seg_model = "medsam"              # Only MedSAM2 is currently supported
        self.undo_radius = 3                 # Remove opposing-sign clicks within this pixel radius
        self.use_previous_mask = True        # Feed the previous mask as SAM2 input

        # ── Grounding / VLM ─────────────────────────────────────────────
        self.grounding_model = "gemma"       # Only gemma is currently supported
        self.use_mask_module = True          # Enable mask overlay in VLM input

        # ── Logging & analysis ──────────────────────────────────────────
        self.save_masks_history = False      # Store per-round masks in the JSON record (large!)
        self.compute_per_round_iou = True    # Compute live IoU per round (requires GT mask)

        # ── Conversation history (Gemma only) ───────────────────────────
        self.max_history_length = 5          # Max past (user+assistant) turns to include
        self.use_history = True              # Enable history in prompts
        self.reset_history_per_image = False # Start a fresh session for each new image

        # ── Paths ───────────────────────────────────────────────────────
        self.output_dir = "./intermediate_results"   # Per-image records & visualizations
        self.results_dir = "./results"               # Aggregated batch evaluation results

        # ── Model checkpoints ───────────────────────────────────────────
        self.model = None                    # Gemma 4 checkpoint path
        self.checkpoint = None               # Generic seg checkpoint (unused)
        self.seg_checkpoint = None            # MedSAM2 checkpoint (.pt)
        self.seg_config = None                # MedSAM2 config YAML


# ═══════════════════════════════════════════════════════════════════════════
#  Single-image inference orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class SingleImageInference:
    """Orchestrates the interactive VLM-guided segmentation for one image.

    The inference loop:
      1. Send the image + target description to the VLM (Gemma 4).
      2. Parse the VLM's tool-call response (add_bbox, add_point, stop_action).
      3. Feed the tool arguments to MedSAM2 to produce/update the mask.
      4. Overlay the mask on the image and send it back to the VLM.
      5. Repeat until stop_action or max_clicks.
    """

    def __init__(self, grounding_model, segmentation_model, args):
        """
        Args:
            grounding_model:    VLM instance (GroundingModel_Gemma_WithHistory).
            segmentation_model: Segmentation model instance (SAMModel / MedSAM2).
            args:               InferenceArgs configuration object.
        """
        self.grounding_model = grounding_model
        self.segmentation_model = segmentation_model
        self.args = args
        self.workspace = getattr(args, 'output_dir')
        self.dataset_name = getattr(args, 'dataset_name', 'default')

        # ── Detect history support ──────────────────────────────────────
        # If the grounding model has a start_new_session() method, it
        # supports conversation history.  Otherwise we run stateless.
        grounding_model_type = getattr(args, 'grounding_model', '')
        if 'gemma' in grounding_model_type:
            if hasattr(self.grounding_model, 'start_new_session'):
                print("Detected Gemma model, enabling conversation history")
                self.use_history = True
            else:
                print(f"Warning: grounding_model set to {grounding_model_type} "
                      "but model does not support history")
                self.use_history = False
        else:
            self.use_history = False

    # ──────────────────────────────────────────────────────────────────────
    #  Main inference loop
    # ──────────────────────────────────────────────────────────────────────

    def forward_single_image(self, img_path: str, target_description: str,
                             max_clicks: int = 5, visualize: bool = True,
                             gt_mask=None) -> tuple:
        """Run the multi-round interactive segmentation loop for one image.

        Args:
            img_path:             Path to the input medical image.
            target_description:   Text prompt describing what to segment.
            max_clicks:           Maximum number of VLM tool-call rounds.
            visualize:            Save per-step visualization images.
            gt_mask:              Optional ground-truth mask for live per-round IoU logging.

        Returns:
            (final_mask, inference_record)
                final_mask:         The last predicted mask (torch.Tensor or np.ndarray).
                inference_record:   Dict with full interaction history.
        """
        print(f"Start processing image: {img_path}")
        print(f"Target description: {target_description}")

        # ── Session management for history-enabled models ───────────────
        session_id = None
        if self.use_history:
            img_name = os.path.basename(img_path).split('.')[0]
            # Derive a deterministic session ID from the image + description
            session_id = f"{img_name}_{hash(target_description) % 10000}"

            if getattr(self.args, 'reset_history_per_image', True):
                # Start a clean conversation for this image
                session_id = self.grounding_model.start_new_session(session_id)
                print(f"Created new conversation session for image: {session_id}")
            else:
                # Reuse or continue an existing session
                self.grounding_model.switch_session(session_id)
                summary = self.grounding_model.get_session_summary()
                print(f"Using session {session_id}, "
                      f"current history length: {summary['current_history_length']} rounds")

        # ── Read image ──────────────────────────────────────────────────
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Cannot read image: {img_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_height, original_width = image.shape[:2]
        self.args.current_image_name = os.path.basename(img_path).split('.')[0]

        # ── Optional resize for VLM input ───────────────────────────────
        # The segmentation model always works at original resolution.
        # The VLM, however, gets a resized version to keep inference fast
        # and within its context window.  We pad to a square with white.
        grounding_resized_img_path = None
        orig_img_path = img_path
        grounding_img_input = orig_img_path
        if hasattr(self.args, 'grounding_resize') and self.args.grounding_resize is not None:
            target_size = self.args.grounding_resize
            print(f"Resize image for grounding model from {original_width}x{original_height} "
                  f"to {target_size}x{target_size}")

            # Letterbox pad to square
            image_pil = Image.fromarray(image_rgb)
            image_pil.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)

            resized_image = Image.new('RGB', (target_size, target_size), (255, 255, 255))
            offset = ((target_size - image_pil.size[0]) // 2,
                      (target_size - image_pil.size[1]) // 2)
            resized_image.paste(image_pil, offset)

            grounding_img_input = resized_image
            print(f"Created in-memory grounding resize image")

        height, width = image.shape[:2]

        # ── Inference record (serialised as JSON at the end) ────────────
        inference_record = {
            "img_path": img_path,
            "target_description": target_description,
            "height": height,
            "width": width,
            "clicks_history": [],
            "model_outputs": [],
            "final_mask": None,
        }

        # ── Ground-truth setup for live per-round metrics ───────────────
        if getattr(self.args, 'compute_per_round_iou', False) and gt_mask is not None:
            inference_record["per_round_metrics"] = []
            # Normalize GT to a binary numpy array
            if isinstance(gt_mask, Image.Image):
                gt_mask_np = np.array(gt_mask)
            else:
                gt_mask_np = gt_mask
            if len(gt_mask_np.shape) == 3:
                gt_mask_np = gt_mask_np[:, :, 0]
            gt_mask_binary = (gt_mask_np > 0).astype(np.uint8)
            print("Enabled real-time per-round IoU computation")
        else:
            gt_mask_binary = None

        # ── Initialise segmentation model with the image ────────────────
        simple_click_image = self.segmentation_model.image_process(img_path=orig_img_path)

        # noinspection PyUnresolvedReferences
        with torch.no_grad():
            self.segmentation_model.set_input_image(simple_click_image)

            clicker = Clicker(
                dataset_name=self.dataset_name,
                output_dir=getattr(self.args, 'output_dir'),
            )
            previous_mask = None
            pred_logits = None
            last_ref_box_str = None
            current_box = None           # bbox from the VLM, carried to later rounds
            retry_count = 0
            max_retries = 1              # One automatic retry on invalid VLM output
            actual_click_count = 0       # Successful rounds (excludes retries)
            click_id = 0                 # Total attempts (including retries)

            print(f"Start multi-round interactive segmentation, max rounds: {max_clicks}")

            # ════════════════════════════════════════════════════════════
            #  Main interactive loop
            # ════════════════════════════════════════════════════════════
            while actual_click_count < max_clicks:
                print(f"\n--- Round {actual_click_count + 1} (Attempt {click_id + 1}) ---")

                # ── Build input structure for the VLM ───────────────────
                inputs = {
                    "img_path": orig_img_path,
                    "caption": [target_description],
                    "height": height,
                    "width": width,
                    "pred_list": [],
                }

                # ── Build the prompt ────────────────────────────────────
                if self.use_history:
                    use_history = getattr(self.args, 'use_history', True)
                    reset_history = (
                        actual_click_count == 0
                        and getattr(self.args, 'reset_history_per_image', True)
                    )
                    prompt, conv = self.grounding_model.build_prompt(
                        inputs,
                        last_ref_box_str,
                        use_history=use_history,
                        reset_history=reset_history,
                    )
                    print(f"Prompt built with history, "
                          f"history length: {len(self.grounding_model.conversation_history) // 2} rounds")
                else:
                    prompt, conv = self.grounding_model.build_prompt(inputs, last_ref_box_str)

                # ── Get VLM response ────────────────────────────────────
                mask_for_prompt = (
                    previous_mask
                    if previous_mask is not None
                    else np.zeros((height, width), dtype=np.uint8)
                )

                if self.use_history:
                    outputs = self.grounding_model.generate_response(
                        prompt, grounding_img_input, mask_for_prompt, conv,
                        save_history=True, round_num=actual_click_count,
                    )
                else:
                    outputs = self.grounding_model.generate_response(
                        prompt, grounding_img_input, mask_for_prompt, conv,
                        round_num=actual_click_count,
                    )

                # Prepend any cached box reference string
                if last_ref_box_str is not None:
                    outputs = last_ref_box_str + outputs

                print(f"Model output: {outputs}")
                inference_record["model_outputs"].append(outputs)

                # Reset box chaining (no longer used in current tool format)
                last_ref_box_str = None

                # ── Parse the VLM's tool call ──────────────────────────
                is_positive, points, should_stop = self.grounding_model.process_response(outputs)

                # ── Stop action ─────────────────────────────────────────
                if should_stop:
                    if actual_click_count == 0 and retry_count < max_retries:
                        print("Model requested stop in first round, retrying once...")
                        retry_count += 1
                        click_id += 1
                        continue
                    print("Model requested stop, ending interaction early")
                    click_info = {"operation": "stop"}
                    self._update_record(inference_record, actual_click_count,
                                        current_box, previous_mask, click_info,
                                        outputs, gt_mask_binary)
                    break

                # ── Bounding box output ─────────────────────────────────
                if is_positive == 'bbox' and points is not None and len(points) == 4:
                    retry_count = 0
                    # Convert from [0, 1] normalized to absolute pixel coords
                    bbox_abs = [
                        int(round(points[0] * width)),   # x1
                        int(round(points[1] * height)),  # y1
                        int(round(points[2] * width)),   # x2
                        int(round(points[3] * height)),  # y2
                    ]
                    print(f"Generated bbox - normalized: [{points[0]:.3f}, "
                          f"{points[1]:.3f}, {points[2]:.3f}, {points[3]:.3f}]")
                    print(f"Generated bbox - absolute: [{bbox_abs[0]}, "
                          f"{bbox_abs[1]}, {bbox_abs[2]}, {bbox_abs[3]}]")

                    current_box = bbox_abs
                    print("Saved bbox for later rounds")

                    # Segment with the bbox directly (no point clicks)
                    pred_mask = self.segmentation_model.get_prediction(
                        clicker=None, box=bbox_abs, mask=pred_logits
                    )

                    if isinstance(pred_mask, tuple):
                        pred_mask, pred_logits = pred_mask

                    previous_mask = (
                        torch.from_numpy(pred_mask).to(torch.uint8).unsqueeze(0)
                    )

                    click_info = {"operation": "add_bbox", "bbox": bbox_abs}
                    self._update_record(inference_record, actual_click_count,
                                        bbox_abs, previous_mask, click_info,
                                        outputs, gt_mask_binary)

                    actual_click_count += 1
                    click_id += 1

                # ── Point output ────────────────────────────────────────
                elif is_positive is not None and points is not None:
                    retry_count = 0

                    # Convert normalized (x, y) to absolute pixel coords
                    abs_x = round(points[0] * width)
                    abs_y = round(points[1] * height)
                    abs_points = (abs_y, abs_x)   # Clicker expects (y, x)

                    print(f"Generated click point - normalized: "
                          f"({points[0]:.3f}, {points[1]:.3f})")
                    print(f"Generated click point - absolute: x={abs_x}, "
                          f"y={abs_y}, label: {is_positive}")

                    click = Click(is_positive=is_positive, coords=abs_points)
                    clicker.add_click(click, getattr(self.args, 'undo_radius', 3))

                    # Box strategy:
                    #   - If we have a cached box and no logits yet, use box+points
                    #   - If we already have logits, use points only (iterative refinement)
                    box_to_use = current_box if pred_logits is None else None

                    if pred_logits is None and current_box is not None:
                        print(f"Segment with box + points, box: {current_box}, "
                              f"points: {len(clicker.clicks_list)}")
                    elif pred_logits is not None:
                        print(f"Refinement with points + previous mask, "
                              f"points: {len(clicker.clicks_list)}")
                    else:
                        print(f"First round, segment with points only, "
                              f"points: {len(clicker.clicks_list)}")

                    pred_mask = self.segmentation_model.get_prediction(
                        clicker, box=box_to_use, mask=pred_logits
                    )

                    if isinstance(pred_mask, tuple):
                        pred_mask, pred_logits = pred_mask

                    previous_mask = (
                        torch.from_numpy(pred_mask).to(torch.uint8).unsqueeze(0)
                    )

                    click_info = {
                        "operation": "add_point",
                        "is_positive": is_positive,
                        "coords": abs_points,
                    }
                    self._update_record(inference_record, actual_click_count,
                                        current_box, previous_mask, click_info,
                                        outputs, gt_mask_binary)

                    actual_click_count += 1
                    click_id += 1

                # ── Invalid output ──────────────────────────────────────
                else:
                    retry_count += 1
                    print(f"Invalid model output (no points and no stop): {outputs}")
                    print(f"Current retries: {retry_count}/{max_retries}")

                    if retry_count > max_retries:
                        print(f"Reached max retries ({max_retries}), stopping inference")
                        break
                    else:
                        print("Retrying with the same prompt...")
                        click_id += 1
                        continue

                # ── Per-round visualization ─────────────────────────────
                if visualize and (len(clicker) > 0 or current_box is not None):
                    try:
                        point_list = [click.coords for click in clicker.clicks_list]
                        label_list = [click.is_positive for click in clicker.clicks_list]
                        self._save_pointlist_visualization(
                            image_rgb, previous_mask, point_list, label_list,
                            img_path, actual_click_count - 1, outputs, current_box,
                        )
                    except Exception as e:
                        print(f"Failed to save point list visualization: {e}")

            # ── End of loop summary ─────────────────────────────────────
            print(f"\nInference loop finished:")
            print(f"  - Actual valid rounds: {actual_click_count}")
            print(f"  - Total attempts: {click_id}")
            print(f"  - Reached max rounds: {'Yes' if actual_click_count >= max_clicks else 'No'}")

            # Add session info to the record
            if self.use_history and session_id:
                summary = self.grounding_model.get_session_summary()
                inference_record["session_info"] = {
                    "session_id": session_id,
                    "total_history_turns": summary["current_history_length"],
                    "max_history_length": summary["max_history_length"],
                }
                print(f"Session {session_id} completed, "
                      f"saved {summary['current_history_length']} history rounds")

            print(f"\nInference complete, "
                  f"{len(inference_record['clicks_history'])} rounds of interaction")

            return previous_mask, inference_record

    # ──────────────────────────────────────────────────────────────────────
    #  Record-keeping helpers
    # ──────────────────────────────────────────────────────────────────────

    def _update_record(self, record: dict, click_id: int, box, mask,
                       click_info, outputs: str, gt_mask_binary=None) -> None:
        """Append the current round's data to the inference record.

        Optionally stores per-round masks and computes live IoU/Dice if a
        ground-truth mask is available.
        """
        # ── Save per-round mask (enabled via save_masks_history flag) ───
        if getattr(self.args, 'save_masks_history', False) and mask is not None:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze(0).cpu().numpy()
            else:
                mask_np = mask
            if "masks_history" not in record:
                record["masks_history"] = []
            record["masks_history"].append(mask_np.tolist())

        # ── Live per-round IoU / Dice ───────────────────────────────────
        if (getattr(self.args, 'compute_per_round_iou', False)
                and gt_mask_binary is not None and mask is not None):
            if isinstance(mask, torch.Tensor):
                mask_np = mask.squeeze(0).cpu().numpy()
            else:
                mask_np = mask

            # Resize mask to match GT dimensions if needed
            if mask_np.shape != gt_mask_binary.shape:
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_resized = mask_pil.resize(
                    (gt_mask_binary.shape[1], gt_mask_binary.shape[0]), Image.LANCZOS,
                )
                mask_np = (np.array(mask_resized) > 128).astype(np.uint8)

            dice, iou = get_metrics(mask_np, gt_mask_binary)

            if "per_round_metrics" not in record:
                record["per_round_metrics"] = []
            operation_type = (
                click_info.get('operation', 'unknown')
                if isinstance(click_info, dict) else 'unknown'
            )
            record["per_round_metrics"].append({
                "round": click_id,
                "operation": operation_type,
                "iou": float(iou),
                "dice": float(dice),
            })
            print(f"  Round {click_id} ({operation_type}) - IoU: {iou:.4f}, Dice: {dice:.4f}")

        # ── Click history ───────────────────────────────────────────────
        record["clicks_history"].append({
            "click_id": click_id,
            "click_info": click_info,
            "used_box": box,
            "outputs": outputs,
        })

    # ──────────────────────────────────────────────────────────────────────
    #  Visualization helpers
    # ──────────────────────────────────────────────────────────────────────

    def _save_visualization(self, image, mask, point, img_path, click_id,
                            stage, caption, is_positive=True):
        """Legacy single-point visualization — currently disabled."""
        pass

    def _save_box_only_visualization(self, image, mask, box, img_path,
                                     click_id, caption):
        """Legacy box-only visualization — currently disabled."""
        pass

    def _save_pointlist_visualization(self, image, mask, point_list,
                                      label_list, img_path, click_id,
                                      outputs, box) -> None:
        """Save a visualization of the current mask, all click points, and optional box.

        The image is written to <output_dir>/<image_name>/step_<click_id:02d>.jpg.
        """
        try:
            sample_output_dir = getattr(self.args, 'sample_output_dir', None)
            if sample_output_dir:
                vis_dir = sample_output_dir
            else:
                img_name = os.path.basename(img_path).split('.')[0]
                vis_dir = os.path.join(self.workspace, img_name)
            os.makedirs(vis_dir, exist_ok=True)
            save_path = os.path.join(vis_dir, f"step_{click_id:02d}.jpg")

            visualize_mask_and_pointlist(
                image, mask, point_list, save_path, label_list, outputs, box,
            )
        except Exception as e:
            print(f"Failed to save point list visualization: {e}")

    # ──────────────────────────────────────────────────────────────────────
    #  Record persistence
    # ──────────────────────────────────────────────────────────────────────

    def save_inference_record(self, record: dict, save_path: str = None) -> str:
        """Save the inference record as a JSON file (without raw mask arrays).

        Args:
            record:    The inference_record dict from forward_single_image().
            save_path: Optional explicit path.  Defaults to
                       <output_dir>/<image_name>/inference_record.json.

        Returns:
            The absolute path to the saved JSON.
        """
        if save_path is None:
            sample_output_dir = getattr(self.args, 'sample_output_dir', None)
            if sample_output_dir:
                save_path = os.path.join(sample_output_dir, "inference_record.json")
            else:
                img_name = os.path.basename(record["img_path"]).split('.')[0]
                save_path = os.path.join(self.workspace, img_name, "inference_record.json")

        # Strip out large mask arrays — keep only structured data
        record_simplified = {
            "img_path": record["img_path"],
            "target_description": record["target_description"],
            "height": record["height"],
            "width": record["width"],
            "num_clicks": len(record["clicks_history"]),
            "clicks_history": record["clicks_history"],
            "model_outputs": record["model_outputs"],
            "per_round_metrics": record.get("per_round_metrics", []),
            "session_info": record.get("session_info", {}),
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(record_simplified, f, indent=2, ensure_ascii=False)

        return save_path


# ═══════════════════════════════════════════════════════════════════════════
#  Convenience entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_single_image_inference(img_path: str, target_description: str,
                               grounding_model, segmentation_model, args,
                               max_clicks: int = 5) -> tuple:
    """Convenience function: instantiate an inferencer, run it, save the record.

    Args:
        img_path:            Path to the input image.
        target_description:  Text description of the target.
        grounding_model:     VLM grounding model instance.
        segmentation_model:  Segmentation model instance.
        args:                InferenceArgs object.
        max_clicks:          Maximum click rounds.

    Returns:
        (final_mask, record_path)
    """
    inferencer = SingleImageInference(grounding_model, segmentation_model, args)

    final_mask, record = inferencer.forward_single_image(
        img_path=img_path,
        target_description=target_description,
        max_clicks=max_clicks,
        visualize=True,
    )

    record_path = inferencer.save_inference_record(record)
    return final_mask, record_path
