import os
import sys
import tempfile
from pathlib import Path

import gradio as gr
from PIL import Image

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Adjust imports to match your project structure
from infer.unibiomed_inference_toolusing import InferenceArgs, run_single_image_inference
from infer.models.model_loader import load_model

# We will cache the model variables globally so they are loaded only once in Colab.
SEGMENTATION_MODEL = None
GROUNDING_MODEL = None

def initialize_models(gemma_path, sam_checkpoint, sam_config, grounding_resize, history_len):
    """Load and cache models into global variables."""
    global SEGMENTATION_MODEL, GROUNDING_MODEL
    
    print("Loading models... This may take a moment.")
    args = InferenceArgs()
    args.model_path = gemma_path
    args.seg_checkpoint = sam_checkpoint
    args.seg_config = sam_config
    args.grounding_model = "gemma"
    args.seg_model = "medsam"
    args.grounding_resize = None if grounding_resize == 0 else grounding_resize
    args.max_history_length = history_len

    SEGMENTATION_MODEL, GROUNDING_MODEL = load_model(args)
    return "Models loaded successfully!"

def predict(image, target_description, n_clicks):
    """Run inference on a single image and return the mask/overlay."""
    if SEGMENTATION_MODEL is None or GROUNDING_MODEL is None:
        raise gr.Error("Models are not loaded yet! Please provide paths and click 'Load Models'.")
        
    if image is None:
        raise gr.Error("Please upload an image.")

    # Save the uploaded PIL Image to a temporary path, since the CLI script expects a path
    _, ext = os.path.splitext(getattr(image, 'filename', '.png'))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        image.save(tmp_path)

    try:
        args = InferenceArgs()
        args.n_clicks = int(n_clicks)
        args.grounding_model = "gemma"
        args.seg_model = "medsam"
        args.output_dir = str(PROJECT_ROOT / "infer" / "gradio_temp" / "intermediate")
        args.results_dir = str(PROJECT_ROOT / "infer" / "gradio_temp" / "results")
        
        final_mask, record_path = run_single_image_inference(
            img_path=tmp_path,
            target_description=target_description,
            grounding_model=GROUNDING_MODEL,
            segmentation_model=SEGMENTATION_MODEL,
            args=args,
            max_clicks=args.n_clicks
        )
        
        if final_mask is None:
            return image, "Inference failed or returned no mask."

        # The mask is returned. Depending on the return type, 
        # (assuming it's a numpy array mask), we can blend it into the original image.
        import numpy as np
        
        # Simple color blend if final_mask is a 2D array of {0, 1}
        img_np = np.array(image).astype(np.float32)
        mask_expanded = np.expand_dims(final_mask > 0, axis=-1)
        
        # Red overlay
        overlay = img_np.copy()
        overlay[final_mask > 0] = overlay[final_mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5
        overlay = overlay.astype(np.uint8)
        
        result_img = Image.fromarray(overlay)
        return result_img, f"Inference complete! Record saved at: {record_path}"
        
    except Exception as e:
        return image, f"Error: {e}"
    finally:
        os.remove(tmp_path)


# Build the Gradio interface
with gr.Blocks(title="GemmaSAM Inference") as demo:
    gr.Markdown("# GemmaSAM Single Inference")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Model Configuration")
            gemma_path = gr.Textbox(label="Gemma Checkpoint Dir", placeholder="/path/to/gemma/model")
            sam_checkpoint = gr.Textbox(label="MedSAM2 Checkpoint Path", placeholder="/path/to/sam2.pt")
            sam_config = gr.Textbox(label="MedSAM2 Config Path", placeholder="sam2_hiera_t.yaml")
            grounding_resize = gr.Number(label="Grounding Resize Resolution (0 for None)", value=512)
            history_len = gr.Number(label="Max History Length", value=5)
            load_btn = gr.Button("Load Models", variant="primary")
            load_status = gr.Textbox(label="Load Status", interactive=False)
            
            load_btn.click(
                initialize_models,
                inputs=[gemma_path, sam_checkpoint, sam_config, grounding_resize, history_len],
                outputs=[load_status]
            )
            
        with gr.Column():
            gr.Markdown("### 2. Inference")
            inp_img = gr.Image(type="pil", label="Upload Image")
            target_desc = gr.Textbox(label="Target Description", placeholder="e.g., right kidney in abdomen CT")
            n_clicks = gr.Slider(minimum=1, maximum=20, step=1, value=5, label="Max Clicks")
            
            infer_btn = gr.Button("Run Inference", variant="primary")
            
            out_img = gr.Image(type="pil", label="Result Image")
            out_status = gr.Textbox(label="Inference Status", interactive=False)
            
            infer_btn.click(
                predict,
                inputs=[inp_img, target_desc, n_clicks],
                outputs=[out_img, out_status]
            )

if __name__ == "__main__":
    # If users run on colab, share=True exposes a public URL 
    demo.launch(share=True)
