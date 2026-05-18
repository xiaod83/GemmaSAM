"""
Model loading and MedSAM2 segmentation wrapper for GemmaSAM.

Provides:
  - SAMModel:      MedSAM2 wrapper built on the sam2 library (SAM2ImagePredictor).
  - SegmentationModel:  Legacy generic wrapper (minimal, delegates to a predictor).
  - load_model():  One-call loader for both grounding + segmentation models.

The grounding model (Gemma 4 with conversation history) is defined in
unibiomed_inference_toolusing.py and imported here.
"""

import torch
from pathlib import Path
import sys
import os
import re
import json

# ── Path setup ──────────────────────────────────────────────────────────────
# Ensure the third_party/sam2 directory is importable so we can do
#   from sam2.build_sam import build_sam2
sys.path.append('../')
project_root = Path(__file__).resolve().parents[2]
sam2_root = project_root / "third_party" / "sam2"
if str(sam2_root) not in sys.path:
    sys.path.insert(0, str(sam2_root))

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ── Imports ─────────────────────────────────────────────────────────────────
import os
import cv2
import time
import numpy as np
from PIL import Image
from utils.clicker import Clicker
from utils.visual_utils import (
    visualize_mask_and_point,
    overlay_points,
    overlay_boxes,
    visualize_mask_and_pointlist,
    overlay_mask,
)
from PIL import Image, ImageOps
from transformers import AutoProcessor, AutoModelForImageTextToText
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ═══════════════════════════════════════════════════════════════════════════
#  Utility helpers
# ═══════════════════════════════════════════════════════════════════════════

def overlay_mask(image_array, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay a semi-transparent coloured mask on an image array.

    Args:
        image_array: (H, W, 3) numpy array (RGB).
        mask:        Binary mask (H, W).
        color:       RGB colour tuple.
        alpha:       Transparency factor (0 = transparent, 1 = solid).

    Returns:
        (H, W, 3) uint8 array with the mask overlaid.
    """
    mask = (mask > 0)
    overlay = image_array.copy()
    overlay[mask] = image_array[mask] * (1 - alpha) + np.array(color) * alpha
    return np.clip(overlay, 0, 255).astype('uint8')


def ensure_intermediate_results_dir(output_dir=None):
    """Create and return an output directory, defaulting to <script_dir>/intermediate_results."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        intermediate_dir = os.path.join(project_root, 'intermediate_results')
        os.makedirs(intermediate_dir, exist_ok=True)
        return intermediate_dir


# ── Colour palette for visualisation ──────────────────────────────────────
color_map = {
    'green': [0, 255, 0],
    'red': [255, 0, 0],
    'blue': [0, 0, 255],
}


# ═══════════════════════════════════════════════════════════════════════════
#  Tool definitions & system prompt for Gemma 4 (VLM grounding model)
# ═══════════════════════════════════════════════════════════════════════════
#
# The VLM is prompted to call one of three tools via XML-encoded JSON:
#
#   <tool_call>
#   {"name": "add_bbox",      "arguments": {"bbox_2d": [x1, y1, x2, y2]}}
#   </tool_call>
#   <tool_call>
#   {"name": "add_point",     "arguments": {"point_2d": [x, y], "point_type": "positive"}}
#   </tool_call>
#   <tool_call>
#   {"name": "stop_action",   "arguments": {}}
#   </tool_call>
#
# Coordinates are in [0, 999] range and are normalised to [0, 1] during parsing.

tools_json_v2 = [
    {
        "type": "function",
        "function": {
            "name": "add_bbox",
            "description": "Add a bounding box to initialize or refine the segmentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bbox_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                        "description": "2D bounding box in [x1, y1, x2, y2] format",
                    }
                },
                "required": ["bbox_2d"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_point",
            "description": (
                "Add a point to refine the mask "
                "(positive to include areas, negative to exclude areas)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "point_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "2D coordinate point in [x, y] format, "
                                       "with x and y in range [0, 999]",
                    },
                    "point_type": {
                        "type": "string",
                        "enum": ["positive", "negative"],
                        "description": "Type of point: 'positive' to expand mask, "
                                       "'negative' to refine mask",
                    },
                },
                "required": ["point_2d", "point_type"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_action",
            "description": "Stop the refinement process when the mask accurately covers the target object.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    },
]

# ── System prompt instructing the VLM on its role and output format ───────
system_prompt_v2 = (
    "You are a professional segmentation annotator specializing in mask creation and refinement. "
    "Your core task is to segment the USER-SPECIFIED TARGET REGION from the provided image. "
    "No preliminary mask is available\u2014you must first create an initial mask using the tool, "
    "then iteratively refine it to achieve pixel-level accuracy. "
    "The mask will be displayed as a semi-transparent green overlay; your goal is to ensure "
    "it exactly covers the entire target region and excludes all non-target areas "
    "(e.g., background, adjacent objects).\n\n"
    "# Tools\n\n"
    "You must call one function to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    f"{chr(10).join([json.dumps(tool) for tool in tools_json_v2])}\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments "
    "within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "Only use the provided functions to complete your task. "
    "Do not invent or assume any other functions. "
    "Carefully consider the current mask state before each action."
)

# User prompts for each turn
prompt_turn_1 = (
    "<image>The target to be segmented is: {target_description}.\n"
    "Now, please analyze the original image, then decide your first action."
)
prompt_later_turn = (
    "<image>Here is the updated mask after your previous action. "
    "Based on this, what is your next action? "
    "If the mask is now accurate, you can call 'stop_action' to finish."
)


# ═══════════════════════════════════════════════════════════════════════════
#  Grounding model — Gemma 4 with conversation history
# ═══════════════════════════════════════════════════════════════════════════

class GroundingModel_Gemma_WithHistory:
    """Wraps Google Gemma 4 as a grounding VLM that outputs segmentation tool calls.

    Key features:
      - Multi-session conversation history (one conversation per image).
      - Mask overlay on the image before feeding to the VLM (so it sees progress).
      - Tool-call JSON parsing with support for add_bbox / add_point / stop_action.
      - Automatic cleanup of intermediate images and GPU memory.

    The model is loaded via Hugging Face ``transformers``
    (``AutoModelForImageTextToText`` + ``AutoProcessor``).
    """

    def __init__(self, model_path: str, args):
        """
        Args:
            model_path: Path to the Gemma 4 Hugging Face checkpoint.
            args:       InferenceArgs or similar with fields:
                        output_dir, use_mask_module, visualize, use_fp16,
                        max_history_length, etc.
        """
        output_dir = getattr(args, 'output_dir', None)
        self.workspace = ensure_intermediate_results_dir(output_dir)
        self.use_mask_module = args.use_mask_module
        self.visualize = args.visualize
        self.args = args

        # ── Load Gemma 4 ────────────────────────────────────────────────
        use_fp16 = getattr(args, 'use_fp16', False)
        if use_fp16:
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
                print("Loading model with BF16 precision")
            else:
                dtype = torch.float16
                print("Loading model with FP16 precision")
        else:
            dtype = torch.float32
            print("Loading model with FP32 precision")

        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=dtype, device_map="auto",
            )
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.predictor = None   # Reserved for potential SAM predictor integration

        # ── Session & history management ────────────────────────────────
        self.conversation_history: list = []
        self.max_history_length = getattr(args, 'max_history_length', 10)
        self.current_session_id = None
        self.session_histories: dict[str, list] = {}   # session_id -> history
        self.intermediate_images: dict[str, list] = {} # session_id -> image paths
        self.temp_dirs: set = set()

    # ──────────────────────────────────────────────────────────────────────
    #  Session management
    # ──────────────────────────────────────────────────────────────────────

    def start_new_session(self, session_id: str = None) -> str:
        """Start a new conversation session (fresh history).

        Args:
            session_id: Optional custom ID.  Auto-generated if None.

        Returns:
            The session ID.
        """
        if session_id is None:
            session_id = f"session_{time.time()}"

        self.current_session_id = session_id
        if session_id not in self.session_histories:
            self.session_histories[session_id] = []
        self.conversation_history = self.session_histories[session_id]

        if session_id not in self.intermediate_images:
            self.intermediate_images[session_id] = []

        print(f"Started new conversation session: {session_id}")
        return session_id

    def switch_session(self, session_id: str) -> None:
        """Switch to an existing session, or create it if it doesn't exist."""
        if session_id in self.session_histories:
            self.current_session_id = session_id
            self.conversation_history = self.session_histories[session_id]
            if session_id not in self.intermediate_images:
                self.intermediate_images[session_id] = []
            print(f"Switched to session: {session_id}")
        else:
            print(f"Session {session_id} not found, creating new session")
            self.start_new_session(session_id)

    def get_system_prompt(self) -> str:
        """Return the system prompt that instructs the VLM on tool calling."""
        return system_prompt_v2

    # ──────────────────────────────────────────────────────────────────────
    #  Image loading
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_image_for_message(image_path) -> Image.Image:
        """Load an image as RGB PIL.  Accepts either a path string or a PIL Image."""
        if isinstance(image_path, Image.Image):
            return image_path.convert('RGB')
        return Image.open(image_path).convert('RGB')

    # ──────────────────────────────────────────────────────────────────────
    #  Prompt construction
    # ──────────────────────────────────────────────────────────────────────

    def build_prompt(self, init_inputs: dict, last_ref_box_str: str = None,
                     use_history: bool = True, reset_history: bool = False):
        """Build a Gemma-compatible message list with conversation history.

        The message list follows the Hugging Face chat-template format with
        ``role`` and ``content`` keys.  Images are included as inline content
        blocks (``{"type": "image", "image": ...}``).

        Args:
            init_inputs:    Dict with ``img_path``, ``caption`` (list[str]), etc.
            last_ref_box_str:  Optional string referencing a previous bbox.
            use_history:    Whether to include past conversation turns.
            reset_history:  If True, clear the current session history first.

        Returns:
            (messages, image_path)
                messages:    List of dicts (role + content) for the chat template.
                image_path:  The original image path from init_inputs.
        """
        if reset_history:
            self.clear_current_session_history()

        image_path = init_inputs['img_path']
        caption = init_inputs['caption'][0]

        is_first_turn = len(self.conversation_history) == 0

        if is_first_turn:
            current_text = prompt_turn_1.format(target_description=caption)
        else:
            action_num = len(self.conversation_history) // 2 + 1
            current_text = prompt_later_turn.format(action_num=action_num)

        if last_ref_box_str:
            current_text += f" {last_ref_box_str}"

        messages: list[dict] = []

        # System message
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": self.get_system_prompt()}],
        })

        # ── Inject history turns ────────────────────────────────────────
        if use_history and self.conversation_history:
            history_to_use = self.conversation_history[-(self.max_history_length * 2):]
            session_intermediate_images = self.intermediate_images.get(
                self.current_session_id, []
            )

            for i, hist_msg in enumerate(history_to_use):
                if hist_msg['role'] == 'user':
                    user_turn_index = i // 2
                    intermediate_image_path = (
                        session_intermediate_images[user_turn_index]
                        if user_turn_index < len(session_intermediate_images)
                        else image_path
                    )

                    hist_content = []
                    for content in hist_msg['content']:
                        if content['type'] == 'image':
                            hist_content.append({
                                "type": "image",
                                "image": intermediate_image_path,
                            })
                        else:
                            hist_content.append(content)
                    messages.append({"role": "user", "content": hist_content})

                elif hist_msg['role'] == 'assistant':
                    messages.append({
                        "role": "assistant",
                        "content": hist_msg['content'],
                    })

        # ── Current turn ────────────────────────────────────────────────
        current_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": current_text},
            ],
        }
        messages.append(current_message)

        return messages, image_path

    # ──────────────────────────────────────────────────────────────────────
    #  Response generation
    # ──────────────────────────────────────────────────────────────────────

    def generate_response(self, messages: list, image_path,
                          masks=None, conv=None, save_history: bool = True,
                          round_num: int = 0) -> str:
        """Generate a VLM response given the conversation messages.

        If a mask is available (round > 0), it is overlaid on the image so
        the VLM can see its previous segmentation result.

        Args:
            messages:     Message list from build_prompt().
            image_path:   Original image path or PIL Image.
            masks:        Previous mask tensor (used for overlay).
            conv:         Unused (legacy parameter).
            save_history: Whether to save this turn to conversation history.
            round_num:    Current round index (0 = first round).

        Returns:
            The raw text response from the VLM.
        """
        # ── Load original image ─────────────────────────────────────────
        if isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
            image_name = getattr(self.args, 'current_image_name', 'image')
        else:
            image = Image.open(image_path).convert('RGB')
            image_name = os.path.basename(image_path).split('.')[0]

        # Handle EXIF orientation
        if hasattr(image, 'exif'):
            image = ImageOps.exif_transpose(image)

        # ── Overlay mask on image for VLM (if not first round) ──────────
        vis_image = image
        if masks is not None and round_num > 0:
            image_np = np.array(image)
            h, w = image_np.shape[:2]

            # Convert mask to numpy
            if isinstance(masks, torch.Tensor):
                mask_np = masks[0].cpu().numpy()
            elif isinstance(masks, (list, tuple)):
                mask_np = masks[0]
            else:
                mask_np = masks

            if mask_np.shape != (h, w):
                mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_resized = mask_pil.resize((w, h), Image.NEAREST)
                mask_np = (np.array(mask_resized) > 128).astype(np.uint8)

            mask_img = overlay_mask(image_np, mask_np, color=color_map['green'])
            vis_image = Image.fromarray(mask_img)
            print(f"[Round {round_num}] Overlayed mask on grounding input image (size: {h}x{w})")

        # ── Save intermediate masked image (optional) ───────────────────
        save_intermediate = getattr(self.args, 'save_intermediate', False)
        tmp_path = None
        tmp_image_for_prompt = vis_image

        if round_num > 0 and masks is not None and save_intermediate:
            sample_output_dir = getattr(self.args, 'sample_output_dir', None)
            if sample_output_dir:
                tmp_dir_path = os.path.join(sample_output_dir, 'interactions')
            else:
                tmp_dir_path = os.path.join(self.workspace, image_name)
                self.temp_dirs.add(tmp_dir_path)
            os.makedirs(tmp_dir_path, exist_ok=True)
            tmp_path = os.path.join(tmp_dir_path, f"round_{round_num:02d}_with_mask.png")
            vis_image.save(tmp_path)
            tmp_image_for_prompt = tmp_path
            print(f"[Round {round_num}] Saved masked image to: {tmp_path}")
        elif round_num == 0 and save_intermediate:
            tmp_image_for_prompt = (
                image_path
                if not isinstance(image_path, Image.Image)
                else vis_image
            )

        # ── Track intermediate images per session ───────────────────────
        if self.current_session_id and round_num > 0:
            if self.current_session_id not in self.intermediate_images:
                self.intermediate_images[self.current_session_id] = []
            if tmp_path:
                self.intermediate_images[self.current_session_id].append(tmp_path)
            else:
                self.intermediate_images[self.current_session_id].append(tmp_image_for_prompt)

            # Trim old images beyond history length
            max_imgs = self.max_history_length
            if len(self.intermediate_images[self.current_session_id]) > max_imgs:
                old_path = self.intermediate_images[self.current_session_id].pop(0)
                if isinstance(old_path, str) and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                        print(f"[Gemma] Removed old intermediate image: {old_path}")
                    except Exception:
                        pass

        # ── Substitute the current-turn image with the masked version ───
        updated_messages = []
        for i, msg in enumerate(messages):
            if msg['role'] == 'user':
                updated_content = []
                for content in msg['content']:
                    if content['type'] == 'image':
                        if i == len(messages) - 1:
                            updated_content.append({
                                "type": "image",
                                "image": tmp_image_for_prompt,
                            })
                        else:
                            updated_content.append(content)
                    else:
                        updated_content.append(content)
                updated_messages.append({"role": msg['role'], "content": updated_content})
            else:
                updated_messages.append(msg)

        # ── Resolve all image refs to PIL for the processor ─────────────
        resolved_messages = []
        for msg in updated_messages:
            if msg['role'] in ('user',):
                resolved_content = []
                for content in msg['content']:
                    if content['type'] == 'image':
                        img_ref = content['image']
                        if isinstance(img_ref, str):
                            pil_img = Image.open(img_ref).convert('RGB')
                        elif isinstance(img_ref, Image.Image):
                            pil_img = img_ref.convert('RGB')
                        else:
                            pil_img = img_ref
                        resolved_content.append({"type": "image", "image": pil_img})
                    else:
                        resolved_content.append(content)
                resolved_messages.append({"role": msg['role'], "content": resolved_content})
            else:
                resolved_messages.append(msg)

        # ── Apply chat template & run inference ─────────────────────────
        text = self.processor.apply_chat_template(
            resolved_messages, tokenize=False, add_generation_prompt=True,
        )

        # Collect all images in order
        all_images = []
        for msg in resolved_messages:
            if msg['role'] == 'user':
                for content in msg['content']:
                    if content['type'] == 'image':
                        all_images.append(content['image'])

        inputs = self.processor(
            text=text,
            images=all_images if all_images else None,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to("cuda")

        # ── Generation config ──────────────────────────────────────────
        do_sample = getattr(self.args, 'do_sample', False)
        temperature = getattr(self.args, 'temperature', 1.0) if do_sample else 1.0
        top_p = getattr(self.args, 'top_p', 1.0) if do_sample else 1.0
        top_k = getattr(self.args, 'top_k', 50) if do_sample else 50

        generation_config = {
            'max_new_tokens': 128,
            'do_sample': do_sample,
        }
        if do_sample:
            generation_config['temperature'] = temperature
            generation_config['top_p'] = top_p
            generation_config['top_k'] = top_k

        generated_ids = self.model.generate(**inputs, **generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        response = output_text[0] if output_text else ""

        # ── Save to conversation history ────────────────────────────────
        if save_history and response:
            self._save_conversation_turn(messages, response, image_path, tmp_image_for_prompt)

        return response

    # ──────────────────────────────────────────────────────────────────────
    #  History management
    # ──────────────────────────────────────────────────────────────────────

    def _save_conversation_turn(self, messages, response, original_image_path,
                                 intermediate_image_path) -> None:
        """Save the current user+assistant turn to the conversation history."""
        current_user_message = messages[-1]

        user_message_to_save = {"role": "user", "content": []}
        for content in current_user_message['content']:
            if content['type'] == 'image':
                user_message_to_save['content'].append({
                    "type": "image",
                    "image": intermediate_image_path,
                })
            else:
                user_message_to_save['content'].append(content)

        assistant_message_to_save = {
            "role": "assistant",
            "content": response,
        }

        self.conversation_history.append(user_message_to_save)
        self.conversation_history.append(assistant_message_to_save)

        # Enforce max history length (trim oldest turns)
        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-(self.max_history_length * 2):]

        if self.current_session_id:
            self.session_histories[self.current_session_id] = self.conversation_history

        print(f"Conversation turn saved. "
              f"Current history length: {len(self.conversation_history) // 2} turns")

    # ──────────────────────────────────────────────────────────────────────
    #  Response parsing (tool call extraction)
    # ──────────────────────────────────────────────────────────────────────

    def process_response(self, outputs: str):
        """Parse the VLM's raw text output and extract the tool call.

        The VLM outputs XML-wrapped JSON like::

            <tool_call>{"name": "add_point", "arguments": {...}}</tool_call>

        Supported functions:
            - add_bbox:      returns ('bbox', [x1, y1, x2, y2], False)
            - add_point:     returns (True/False, (x, y), False)
            - stop_action:   returns (None, None, True)

        Coordinates are in [0, 999] range and are normalised to [0, 1].

        Returns:
            (is_positive, relative_coordinates, should_stop)
        """
        import json

        is_positive = None
        relative_coor = None
        should_stop = False

        tool_call_match = re.search(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>', outputs, re.DOTALL
        )

        if tool_call_match:
            try:
                raw_json = tool_call_match.group(1)

                # Fix missing quotes around bare key names (common Gemma output issue)
                fixed_json = re.sub(
                    r'([,{]\s*)([a-zA-Z_]\w*)(\s*:\s*)',
                    r'\1"\2"\3',
                    raw_json,
                )
                if fixed_json != raw_json:
                    print("Fixed JSON format error:")
                    print(f"  Original: {raw_json}")
                    print(f"  Fixed: {fixed_json}")

                tool_call_json = json.loads(fixed_json)
                function_name = tool_call_json.get("name", "")
                arguments = tool_call_json.get("arguments", {})

                print(f"Parsed function call: {function_name}, args: {arguments}")

                # ── add_bbox ──────────────────────────────────────────
                if function_name == "add_bbox":
                    bbox_2d = arguments.get("bbox_2d")
                    if bbox_2d and isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        normalized_bbox = [coord / 999.0 for coord in bbox_2d]
                        is_positive = 'bbox'
                        relative_coor = normalized_bbox
                    else:
                        print(f"Warning: Invalid bbox_2d format: {bbox_2d}")

                # ── add_point ─────────────────────────────────────────
                elif function_name == "add_point":
                    point_type = arguments.get("point_type", "").lower()
                    point_2d = arguments.get("point_2d")

                    if point_type in ["positive", "negative"] and point_2d:
                        is_positive = (point_type == "positive")
                        if isinstance(point_2d, list) and len(point_2d) >= 2:
                            x, y = point_2d[0], point_2d[1]
                            relative_coor = (x / 999.0, y / 999.0)
                        else:
                            print(f"Warning: Invalid point_2d format: {point_2d}")
                    else:
                        print(f"Warning: Invalid point_type or missing point_2d: {arguments}")

                # ── add_positive_point (legacy) ───────────────────────
                elif function_name == "add_positive_point":
                    is_positive = True
                    x = arguments.get("x")
                    y = arguments.get("y")
                    if x is not None and y is not None:
                        relative_coor = (x / 999.0, y / 999.0)
                    else:
                        print(f"Warning: Positive point missing coordinates: {arguments}")

                # ── add_negative_point (legacy) ───────────────────────
                elif function_name == "add_negative_point":
                    is_positive = False
                    x = arguments.get("x")
                    y = arguments.get("y")
                    if x is not None and y is not None:
                        relative_coor = (x / 999.0, y / 999.0)
                    else:
                        print(f"Warning: Negative point missing coordinates: {arguments}")

                # ── stop_action ───────────────────────────────────────
                elif function_name == "stop_action":
                    should_stop = True
                    is_positive = None
                    relative_coor = None

                else:
                    print(f"Warning: Unknown function name: {function_name}")

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse tool_call JSON: {e}")
                print(f"Raw content: {tool_call_match.group(1)}")
        else:
            print(f"Warning: No <tool_call> tag found in output: {outputs}")

        return is_positive, relative_coor, should_stop

    # ──────────────────────────────────────────────────────────────────────
    #  Session cleanup
    # ──────────────────────────────────────────────────────────────────────

    def clear_current_session_history(self) -> None:
        """Delete all intermediate images and history for the current session."""
        if self.current_session_id and self.current_session_id in self.intermediate_images:
            for img_path in self.intermediate_images[self.current_session_id]:
                if isinstance(img_path, str) and os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        print(f"Removed intermediate image: {img_path}")
                    except Exception:
                        pass
            self.intermediate_images[self.current_session_id] = []

        self.conversation_history.clear()
        if self.current_session_id:
            self.session_histories[self.current_session_id] = []
        print("Current session history cleared")

    def clear_all_sessions(self) -> None:
        """Delete intermediate images and history for all sessions."""
        for session_id, img_paths in self.intermediate_images.items():
            for img_path in img_paths:
                if isinstance(img_path, str) and os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        print(f"Removed intermediate image: {img_path}")
                    except Exception:
                        pass

        self.session_histories.clear()
        self.conversation_history.clear()
        self.intermediate_images.clear()
        self.current_session_id = None
        print("All session histories cleared")

    def get_session_summary(self) -> dict:
        """Return a summary dict for the current session.

        Keys: current_session_id, current_history_length, total_sessions,
              max_history_length, intermediate_images_count.
        """
        intermediate_image_count = 0
        if self.current_session_id and self.current_session_id in self.intermediate_images:
            intermediate_image_count = len(self.intermediate_images[self.current_session_id])

        return {
            "current_session_id": self.current_session_id,
            "current_history_length": len(self.conversation_history) // 2,
            "total_sessions": len(self.session_histories),
            "max_history_length": self.max_history_length,
            "intermediate_images_count": intermediate_image_count,
        }

    def _safe_remove_intermediate(self, img_path, log_prefix="Removed intermediate image"):
        """Safely remove an intermediate image file (handles PIL, bytes, data: URIs)."""
        if not img_path:
            return
        if isinstance(img_path, Image.Image):
            return
        if isinstance(img_path, (bytes, str, Path)):
            path_str = (
                img_path.decode("utf-8", errors="ignore")
                if isinstance(img_path, bytes)
                else str(img_path)
            )
            if path_str.startswith("data:"):
                return
            try:
                if os.path.exists(path_str):
                    os.remove(path_str)
                    print(f"{log_prefix}: {path_str}")
            except OSError:
                pass

    def load_session_history(self, history_data: list, session_id: str = None) -> None:
        """Load a pre-existing history list into a new or existing session."""
        if session_id is None:
            session_id = f"loaded_session_{time.time()}"
        self.session_histories[session_id] = history_data
        self.switch_session(session_id)
        print(f"Loaded history into session: {session_id}")

    def cleanup_session_images(self, session_id: str = None) -> None:
        """Manually clean intermediate images for a given session."""
        if session_id is None:
            session_id = self.current_session_id
        if session_id and session_id in self.intermediate_images:
            for img_path in self.intermediate_images[session_id]:
                self._safe_remove_intermediate(img_path, "Cleaned up intermediate image")
            self.intermediate_images[session_id] = []
            print(f"Cleaned up all intermediate images for session: {session_id}")
        self._cleanup_empty_temp_dirs()

    def release_resources(self) -> None:
        """Clean all temp images, delete the model and processor, and clear GPU cache."""
        for session_id, img_paths in self.intermediate_images.items():
            for img_path in img_paths:
                self._safe_remove_intermediate(img_path, "Cleaned up intermediate image")
        self._cleanup_empty_temp_dirs()
        del self.model
        del self.processor
        torch.cuda.empty_cache()

    def _cleanup_empty_temp_dirs(self) -> None:
        """Remove (or recursively delete) tracked temp directories."""
        import shutil
        for temp_dir in list(self.temp_dirs):
            if os.path.exists(temp_dir):
                try:
                    if not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                        print(f"Removed empty temp directory: {temp_dir}")
                    else:
                        shutil.rmtree(temp_dir)
                        print(f"Removed temp directory and contents: {temp_dir}")
                    self.temp_dirs.discard(temp_dir)
                except Exception as e:
                    print(f"Failed to remove temp directory {temp_dir}: {e}")


# ── Global colour map (redefined here for module-level access) ─────────────
color_map = {
    'green': (0, 255, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
}


# ═══════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════

def load_segmentation_model(args):
    """Load the MedSAM2 segmentation model from the checkpoint in *args*."""
    if args.seg_model != 'medsam':
        raise ValueError(
            f"Only seg_model='medsam' (MedSAM2) is supported, got: {args.seg_model}"
        )
    segmentation_model = SAMModel(args)
    return segmentation_model


def load_grounding_model(args):
    """Load the Gemma 4 grounding VLM from the checkpoint in *args*."""
    if 'gemma' in args.grounding_model.lower():
        grounding_model = GroundingModel_Gemma_WithHistory(args.model, args)
    else:
        raise ValueError(f"Unknown grounding model: {args.grounding_model}")
    return grounding_model


def load_model(args):
    """Load both segmentation and grounding models.

    Returns:
        (segmentation_model, grounding_model)
    """
    segmentation_model = load_segmentation_model(args)
    grounding_model = load_grounding_model(args)
    return segmentation_model, grounding_model


# ═══════════════════════════════════════════════════════════════════════════
#  Segmentation model wrappers
# ═══════════════════════════════════════════════════════════════════════════

class SegmentationModel:
    """Generic segmentation model wrapper (delegates to an internal predictor).

    Used primarily for SimpleClick-based workflows where the predictor
    exposes ``set_input_image()`` and ``get_prediction()``.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    def set_input_image(self, image):
        if self.predictor is not None:
            self.predictor.set_input_image(image)

    def get_prediction(self, clicker, box=None, mask=None):
        pred_mask = self.predictor.get_prediction(clicker)
        return pred_mask > 0.49

    def image_process(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def release_resources(self):
        del self.predictor
        torch.cuda.empty_cache()
        self.predictor = None

    def predict_clicks_from_mask(self, target_mask, file_name=None,
                                  object_id=None, click_num=2, pred_thr=0.49):
        """Generate clicks from a target mask (inverse segmentation).

        This is used for SimpleClick: given a ground-truth mask, simulate
        the clicks a user would make to produce it.

        Returns:
            (clicks, predicted_mask)
        """
        predictor = self.predictor
        pred_mask = np.zeros_like(target_mask)
        clicker = Clicker(gt_mask=target_mask)
        for _ in range(click_num):
            clicker.object_id = object_id
            clicker.make_next_click(pred_mask, file_name)
            pred_probs = predictor.get_prediction(clicker)
            pred_mask = pred_probs > pred_thr
        return clicker.get_clicks(), pred_mask


def get_points_nd(clicks_list) -> tuple:
    """Convert a list of Click objects to NumPy point and label arrays.

    Returns:
        (points, labels) suitable for SAM2ImagePredictor.predict().
    """
    points, labels = [], []
    for click in clicks_list:
        h, w = click.coords_and_indx[:2]
        points.append([w, h])
        labels.append(int(click.is_positive))
    return np.array(points), np.array(labels)


class SAMModel:
    """MedSAM2 segmentation model wrapper built on SAM2ImagePredictor.

    Supports three prediction modes:
      1. Points + optional box (iterative click refinement)
      2. Box only (no clicks)
      3. Points + box + previous mask logits

    Always picks the highest-scoring mask from SAM2's multi-mask output.
    """

    def __init__(self, args=None):
        seg_path = getattr(args, 'seg_checkpoint', None)
        model_config = getattr(args, 'sam_config', None)
        if not model_config and args is not None:
            model_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
        if not seg_path or not model_config:
            raise ValueError("seg_checkpoint must be provided via args for MedSAM2")

        self.predictor = SAM2ImagePredictor(build_sam2(model_config, seg_path))
        print(f"MedSAM2 model loaded: {seg_path}")
        self.pred_thres = 0.50

    def set_input_image(self, image) -> None:
        """Set the image for SAM2 to process."""
        if self.predictor is not None:
            self.predictor.set_image(image)

    def get_prediction(self, clicker=None, box=None, mask=None):
        """Run SAM2 prediction with the given prompts.

        Args:
            clicker: Optional Clicker with accumulated clicks.
            box:     Optional bounding box [x1, y1, x2, y2].
            mask:    Optional previous mask logits for iterative refinement.

        Returns:
            (best_mask, best_logits) — the highest-scoring SAM2 output.
        """
        pred_logits = mask

        if clicker is not None:
            if box is not None:
                box = np.array(box)

            if pred_logits is not None and len(clicker.get_clicks()) > 0:
                clicks_list = clicker.get_last_click()
                print(f"[Iterative refinement] Using latest point + previous mask, "
                      f"points: {len(clicks_list)}")
            else:
                clicks_list = clicker.get_clicks()
                print(f"[First round] Using all points, points: {len(clicks_list)}")

            points_nd, labels_nd = get_points_nd(clicks_list)
            masks, scores, logits = self.predictor.predict(
                point_coords=points_nd,
                point_labels=labels_nd,
                mask_input=pred_logits,
                box=box,
            )
            max_score_idx = np.argmax(scores)
            return masks[max_score_idx], logits[[max_score_idx]]

        if box is not None:
            box = np.array(box)
            masks, scores, logits = self.predictor.predict(
                box=box, mask_input=pred_logits
            )
            max_score_idx = np.argmax(scores)
            return masks[max_score_idx], logits[[max_score_idx]]

        raise ValueError("Either clicker or box must be provided for prediction")

    @staticmethod
    def image_process(img_path: str) -> np.ndarray:
        """Read an image from disk and convert to RGB numpy array."""
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def release_resources(self) -> None:
        """Free the predictor and clear GPU cache."""
        del self.predictor
        torch.cuda.empty_cache()
        self.predictor = None
