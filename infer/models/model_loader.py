import torch
from pathlib import Path
import sys
import os
import sys
import re
import json

sys.path.append('../')
project_root = Path(__file__).resolve().parents[2]
# ensure third_party/sam2 is importable
sam2_root = project_root / "third_party" / "sam2"
if str(sam2_root) not in sys.path:
    sys.path.insert(0, str(sam2_root))
# ensure project root is importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# --- Dependency imports ---
import os
import cv2
from utils.clicker import Clicker
import time
import numpy as np
from PIL import Image
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

# Assume you have these helper functions
def overlay_mask(image_array, mask, color=[0, 255, 0], alpha=0.5):
    """Overlay the mask on the image"""
    mask = (mask > 0)
    overlay = image_array.copy()
    overlay[mask] = image_array[mask] * (1 - alpha) + np.array(color) * alpha
    return np.clip(overlay, 0, 255).astype('uint8')

def ensure_intermediate_results_dir(output_dir=None):
    """Ensure the output directory exists"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        intermediate_dir = os.path.join(project_root, 'intermediate_results')
        os.makedirs(intermediate_dir, exist_ok=True)
        return intermediate_dir

# Color map
color_map = {
    'green': [0, 255, 0],
    'red': [255, 0, 0],
    'blue': [0, 0, 255]
}


#####Tool Using Version #####
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
                        "description": "2D bounding box in [x1, y1, x2, y2] format"
                    }
                },
                "required": ["bbox_2d"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_point",
            "description": "Add a point to refine the mask (positive to include areas, negative to exclude areas).",
            "parameters": {
                "type": "object",
                "properties": {
                    "point_2d": {
                        "type": "array",
                        "items": {
                            "type": "integer"
                        },
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "2D coordinate point in [x, y] format, with x and y in range [0, 999]"
                    },
                    "point_type": {
                        "type": "string",
                        "enum": ["positive", "negative"],
                        "description": "Type of point: 'positive' to expand mask, 'negative' to refine mask"
                    }
                },
                "required": ["point_2d", "point_type"]
            }
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
                "required": []
            }
        }
    }
]

system_prompt_v2 = (
    "You are a professional segmentation annotator specializing in mask creation and refinement. Your core task is to segment the USER-SPECIFIED TARGET REGION from the provided image. "
    "No preliminary mask is available—you must first create an initial mask using the tool, then iteratively refine it to achieve pixel-level accuracy. "
    "The mask will be displayed as a semi-transparent green overlay; your goal is to ensure it exactly covers the entire target region and excludes all non-target areas (e.g., background, adjacent objects).\n\n"
    "# Tools\n\n"
    "You must call one function to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n"
    f"{chr(10).join([json.dumps(tool) for tool in tools_json_v2])}\n"
    "</tools>\n\n"
    "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
    "<tool_call>\n"
    '{"name": <function-name>, "arguments": <args-json-object>}\n'
    "</tool_call>\n\n"
    "Only use the provided functions to complete your task. Do not invent or assume any other functions. Carefully consider the current mask state before each action."
)

prompt_turn_1 = "<image>The target to be segmented is: {target_description}.\n Now, please analyze the original image, then decide your first action."

prompt_later_turn = "<image>Here is the updated mask after your previous action. Based on this, what is your next action? If the mask is now accurate, you can call 'stop_action' to finish."


class GroundingModel_Gemma_WithHistory():
    def __init__(self, model_path, args):
        output_dir = getattr(args, 'output_dir', None)
        self.workspace = ensure_intermediate_results_dir(output_dir)
        self.use_mask_module = args.use_mask_module
        self.visualize = args.visualize
        self.args = args

        # Load Gemma 4 model and processor
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

        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto"
        ).eval()

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.predictor = None

        # Conversation history management
        self.conversation_history = []
        self.max_history_length = getattr(args, 'max_history_length', 10)
        self.current_session_id = None
        self.session_histories = {}
        self.intermediate_images = {}
        self.temp_dirs = set()

    def start_new_session(self, session_id=None):
        """Start a new conversation session"""
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

    def switch_session(self, session_id):
        """Switch to the specified conversation session"""
        if session_id in self.session_histories:
            self.current_session_id = session_id
            self.conversation_history = self.session_histories[session_id]
            if session_id not in self.intermediate_images:
                self.intermediate_images[session_id] = []
            print(f"Switched to session: {session_id}")
        else:
            print(f"Session {session_id} not found, creating new session")
            self.start_new_session(session_id)

    def get_system_prompt(self):
        """Get the system prompt"""
        return system_prompt_v2

    def _load_image_for_message(self, image_path):
        """
        Load an image for inclusion in a Gemma message.
        Returns a PIL Image if path is a string, or returns as-is if already a PIL Image.
        """
        if isinstance(image_path, Image.Image):
            return image_path.convert('RGB')
        return Image.open(image_path).convert('RGB')

    def build_prompt(self, init_inputs, last_ref_box_str=None, use_history=True, reset_history=False):
        """
        Build Gemma 4 messages format, with conversation history support.
        Gemma uses the standard HF chat template with inline image content blocks.
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

        messages = []

        # Gemma supports system role via chat template
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": self.get_system_prompt()}]
        })

        # Add history messages
        if use_history and self.conversation_history:
            history_to_use = self.conversation_history[-(self.max_history_length * 2):]
            session_intermediate_images = self.intermediate_images.get(self.current_session_id, [])

            for i, hist_msg in enumerate(history_to_use):
                if hist_msg['role'] == 'user':
                    user_turn_index = i // 2
                    if user_turn_index < len(session_intermediate_images):
                        intermediate_image_path = session_intermediate_images[user_turn_index]
                    else:
                        intermediate_image_path = image_path

                    # Build history user message with image content block
                    hist_content = []
                    for content in hist_msg['content']:
                        if content['type'] == 'image':
                            hist_content.append({
                                "type": "image",
                                "image": intermediate_image_path
                            })
                        else:
                            hist_content.append(content)

                    messages.append({"role": "user", "content": hist_content})
                elif hist_msg['role'] == 'assistant':
                    messages.append({"role": "assistant", "content": hist_msg['content']})

        # Current turn user message
        current_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},  # replaced with masked image in generate_response
                {"type": "text", "text": current_text}
            ]
        }
        messages.append(current_message)

        return messages, image_path

    def generate_response(self, messages, image_path, masks=None, conv=None, save_history=True, round_num=0):
        """
        Generate responses using Gemma 4 via HF AutoModelForImageTextToText.
        """
        # Load original image
        if isinstance(image_path, Image.Image):
            image = image_path.convert('RGB')
            image_name = getattr(self.args, 'current_image_name', 'image')
        else:
            image = Image.open(image_path).convert('RGB')
            image_name = os.path.basename(image_path).split('.')[0]
        if hasattr(image, 'exif'):
            image = ImageOps.exif_transpose(image)

        # If a mask exists and this is not the first turn, overlay the mask on the image
        vis_image = image
        if masks is not None and round_num > 0:
            image_np = np.array(image)
            h, w = image_np.shape[:2]

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
            tmp_image_for_prompt = image_path if not isinstance(image_path, Image.Image) else vis_image

        # Save intermediate image path for current round (for history)
        if self.current_session_id and round_num > 0:
            if self.current_session_id not in self.intermediate_images:
                self.intermediate_images[self.current_session_id] = []
            if tmp_path:
                self.intermediate_images[self.current_session_id].append(tmp_path)
            else:
                self.intermediate_images[self.current_session_id].append(tmp_image_for_prompt)
            max_intermediate_images = self.max_history_length
            if len(self.intermediate_images[self.current_session_id]) > max_intermediate_images:
                old_path = self.intermediate_images[self.current_session_id].pop(0)
                if isinstance(old_path, str) and os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                        print(f"[Gemma] Removed old intermediate image: {old_path}")
                    except:
                        pass

        # Update the last user message's image with the (possibly masked) image
        updated_messages = []
        for i, msg in enumerate(messages):
            if msg['role'] == 'user':
                updated_content = []
                for content in msg['content']:
                    if content['type'] == 'image':
                        if i == len(messages) - 1:
                            # Current turn: use masked/vis image
                            updated_content.append({
                                "type": "image",
                                "image": tmp_image_for_prompt
                            })
                        else:
                            updated_content.append(content)
                    else:
                        updated_content.append(content)
                updated_messages.append({"role": msg['role'], "content": updated_content})
            else:
                updated_messages.append(msg)

        # Resolve all image references to PIL Images for the processor
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

        # Apply chat template and process inputs
        text = self.processor.apply_chat_template(
            resolved_messages, tokenize=False, add_generation_prompt=True
        )

        # Collect all PIL images in order for the processor
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

        # Sampling config
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
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        response = output_text[0] if output_text else ""

        if save_history and response:
            self._save_conversation_turn(messages, response, image_path, tmp_image_for_prompt)

        return response

    def _save_conversation_turn(self, messages, response, original_image_path, intermediate_image_path):
        """Save the current conversation turn to history"""
        current_user_message = messages[-1]

        user_message_to_save = {"role": "user", "content": []}
        for content in current_user_message['content']:
            if content['type'] == 'image':
                user_message_to_save['content'].append({
                    "type": "image",
                    "image": intermediate_image_path
                })
            else:
                user_message_to_save['content'].append(content)

        assistant_message_to_save = {
            "role": "assistant",
            "content": response
        }

        self.conversation_history.append(user_message_to_save)
        self.conversation_history.append(assistant_message_to_save)

        if len(self.conversation_history) > self.max_history_length * 2:
            self.conversation_history = self.conversation_history[-(self.max_history_length * 2):]

        if self.current_session_id:
            self.session_histories[self.current_session_id] = self.conversation_history

        print(f"Conversation turn saved. Current history length: {len(self.conversation_history) // 2} turns")

    def process_response(self, outputs):
        """
        Parse model output: extract tool-calling function calls.
        Supports:
            - <tool_call>{"name": "add_point", "arguments": {"point_2d": [x, y], "point_type": "positive"}}</tool_call>
            - <tool_call>{"name": "add_bbox", "arguments": {"bbox_2d": [x1, y1, x2, y2]}}</tool_call>
            - <tool_call>{"name": "stop_action", "arguments": {}}</tool_call>
        Coordinate range: 0-999 (relative coordinate system)
        """
        import json

        is_positive = None
        relative_coor = None
        should_stop = False

        tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', outputs, re.DOTALL)

        if tool_call_match:
            try:
                raw_json = tool_call_match.group(1)

                # Fix common JSON format errors: missing quotes around key names
                fixed_json = re.sub(r'([,{]\s*)([a-zA-Z_]\w*)(\s*":\s*)', r'\1"\2\3', raw_json)
                if fixed_json != raw_json:
                    print("Fixed JSON format error:")
                    print(f"  Original: {raw_json}")
                    print(f"  Fixed: {fixed_json}")

                tool_call_json = json.loads(fixed_json)
                function_name = tool_call_json.get("name", "")
                arguments = tool_call_json.get("arguments", {})

                print(f"Parsed function call: {function_name}, args: {arguments}")

                if function_name == "add_bbox":
                    bbox_2d = arguments.get("bbox_2d")
                    if bbox_2d and isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                        normalized_bbox = [coord / 999.0 for coord in bbox_2d]
                        is_positive = 'bbox'
                        relative_coor = normalized_bbox
                    else:
                        print(f"Warning: Invalid bbox_2d format: {bbox_2d}")

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

                # Legacy format: add_positive_point
                elif function_name == "add_positive_point":
                    is_positive = True
                    x = arguments.get("x")
                    y = arguments.get("y")
                    if x is not None and y is not None:
                        relative_coor = (x / 999.0, y / 999.0)
                    else:
                        print(f"Warning: Positive point missing coordinates: {arguments}")

                # Legacy format: add_negative_point
                elif function_name == "add_negative_point":
                    is_positive = False
                    x = arguments.get("x")
                    y = arguments.get("y")
                    if x is not None and y is not None:
                        relative_coor = (x / 999.0, y / 999.0)
                    else:
                        print(f"Warning: Negative point missing coordinates: {arguments}")

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

    def clear_current_session_history(self):
        """Clear conversation history for current session"""
        if self.current_session_id and self.current_session_id in self.intermediate_images:
            for img_path in self.intermediate_images[self.current_session_id]:
                if isinstance(img_path, str) and os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        print(f"Removed intermediate image: {img_path}")
                    except:
                        pass
            self.intermediate_images[self.current_session_id] = []

        self.conversation_history.clear()
        if self.current_session_id:
            self.session_histories[self.current_session_id] = []
        print("Current session history cleared")

    def clear_all_sessions(self):
        """Clear history for all sessions"""
        for session_id, img_paths in self.intermediate_images.items():
            for img_path in img_paths:
                if isinstance(img_path, str) and os.path.exists(img_path):
                    try:
                        os.remove(img_path)
                        print(f"Removed intermediate image: {img_path}")
                    except:
                        pass

        self.session_histories.clear()
        self.conversation_history.clear()
        self.intermediate_images.clear()
        self.current_session_id = None
        print("All session histories cleared")

    def get_session_summary(self):
        """Get summary info for the current session"""
        intermediate_image_count = 0
        if self.current_session_id and self.current_session_id in self.intermediate_images:
            intermediate_image_count = len(self.intermediate_images[self.current_session_id])

        return {
            "current_session_id": self.current_session_id,
            "current_history_length": len(self.conversation_history) // 2,
            "total_sessions": len(self.session_histories),
            "max_history_length": self.max_history_length,
            "intermediate_images_count": intermediate_image_count
        }

    def _safe_remove_intermediate(self, img_path, log_prefix="Removed intermediate image"):
        if not img_path:
            return
        if isinstance(img_path, Image.Image):
            return
        if isinstance(img_path, (bytes, str, Path)):
            path_str = img_path.decode("utf-8", errors="ignore") if isinstance(img_path, bytes) else str(img_path)
            if path_str.startswith("data:"):
                return
            try:
                if os.path.exists(path_str):
                    os.remove(path_str)
                    print(f"{log_prefix}: {path_str}")
            except OSError:
                pass

    def load_session_history(self, history_data, session_id=None):
        """Load session history"""
        if session_id is None:
            session_id = f"loaded_session_{time.time()}"

        self.session_histories[session_id] = history_data
        self.switch_session(session_id)
        print(f"Loaded history into session: {session_id}")

    def cleanup_session_images(self, session_id=None):
        """Manually clean intermediate images for a specific session"""
        if session_id is None:
            session_id = self.current_session_id

        if session_id and session_id in self.intermediate_images:
            for img_path in self.intermediate_images[session_id]:
                self._safe_remove_intermediate(img_path, "Cleaned up intermediate image")
            self.intermediate_images[session_id] = []
            print(f"Cleaned up all intermediate images for session: {session_id}")

        self._cleanup_empty_temp_dirs()

    def release_resources(self):
        """Release resources"""
        for session_id, img_paths in self.intermediate_images.items():
            for img_path in img_paths:
                self._safe_remove_intermediate(img_path, "Cleaned up intermediate image")

        self._cleanup_empty_temp_dirs()

        del self.model
        del self.processor
        torch.cuda.empty_cache()

    def _cleanup_empty_temp_dirs(self):
        """Clean empty temp folders"""
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


color_map = {
    'green': (0, 255, 0),
    'red': (255, 0, 0),
    'blue': (0, 0, 255)
}


def load_segmentation_model(args):
    """Loads the appropriate segmentation model based on arguments."""
    if args.seg_model != 'medsam':
        raise ValueError(f"Only seg_model='medsam' (MedSAM2) is supported, got: {args.seg_model}")

    segmentation_model = SAMModel(args)
    return segmentation_model


def load_grounding_model(args):
    """Loads the appropriate grounding model based on arguments."""
    if 'gemma' in args.grounding_model.lower():
        grounding_model = GroundingModel_Gemma_WithHistory(args.model, args)
    else:
        raise ValueError(f"Unknown grounding model: {args.grounding_model}")

    return grounding_model


def load_model(args):
    segmentation_model = load_segmentation_model(args)
    grounding_model = load_grounding_model(args)
    return segmentation_model, grounding_model


class SegmentationModel:
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

    def predict_clicks_from_mask(self, target_mask, file_name=None, object_id=None, click_num=2, pred_thr=0.49):
        """
        This function is only used for SimpleClick,
        which is used for predicting initial clicks from a generated mask.
        This is an inverse process.
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


def get_points_nd(clicks_list):
    points, labels = [], []
    for click in clicks_list:
        h, w = click.coords_and_indx[:2]
        points.append([w, h])
        labels.append(int(click.is_positive))
    return np.array(points), np.array(labels)


class SAMModel:
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

    def set_input_image(self, image):
        if self.predictor is not None:
            self.predictor.set_image(image)

    def get_prediction(self, clicker=None, box=None, mask=None):
        pred_logits = mask
        if clicker is not None:
            if box is not None:
                box = np.array(box)
            if pred_logits is not None and len(clicker.get_clicks()) > 0:
                clicks_list = clicker.get_last_click()
                print(f"[Iterative refinement] Using latest points + previous mask, points: {len(clicks_list)}")
            else:
                clicks_list = clicker.get_clicks()
                print(f"[First round] Using all points, points: {len(clicks_list)}")

            points_nd, labels_nd = get_points_nd(clicks_list)
            masks, scores, logits = self.predictor.predict(
                point_coords=points_nd, point_labels=labels_nd, mask_input=pred_logits, box=box)
            max_score_idx = np.argmax(scores)
            return (masks[max_score_idx], logits[[max_score_idx]])

        if box is not None:
            box = np.array(box)
            masks, scores, logits = self.predictor.predict(
                box=box, mask_input=pred_logits)
            max_score_idx = np.argmax(scores)
            return (masks[max_score_idx], logits[[max_score_idx]])

        raise ValueError("Either clicker or box must be provided for prediction")

    def image_process(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def release_resources(self):
        del self.predictor
        torch.cuda.empty_cache()
        self.predictor = None