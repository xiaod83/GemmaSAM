# GemmaSAM


GemmaSAM connects **Google Gemma 4** (a vision-language model) with **MedSAM2** (a medical image segmentation model) to perform interactive, text-guided segmentation of medical images. The VLM acts as a "grounding" agent: it inspects the image, decides where to click or draw a bounding box, and refines the segmentation mask over multiple rounds until the target anatomy is accurately segmented.

This project was created for the **Gemma 4 Good Hackathon** (Google x Kaggle).

---

## Architecture

```
User (target description, e.g. "liver")
        |
        v
+---------------------------+
|  Gemma 4 VLM (grounding)  |  <-- decides where to click/box
|  - inspects image + mask  |
|  - calls tools via JSON   |
+---------------------------+
        |  tool calls: add_bbox, add_point, stop_action
        v
+---------------------------+
|  MedSAM2 (segmentation)   |  <-- produces masks from points/boxes
|  - SAM2.1-based predictor |
+---------------------------+
        |
        v
  Segmentation mask output
```

The inference loop works as follows:

1. **User provides** an image path and a text description of the target (e.g., "liver")
2. **Round 1**: Gemma 4 analyzes the original image and calls `add_bbox` (bounding box) or `add_point` (click point) to initialize the segmentation
3. **MedSAM2** generates a mask from the VLM's prompt
4. **Subsequent rounds**: Gemma 4 sees the mask overlaid on the image, and calls more points or a bbox to refine it
5. **Termination**: Gemma 4 calls `stop_action` when the mask is accurate, or the max click limit is reached
6. **Result**: Final segmentation mask and a JSON inference record are saved

---

## File Structure

```
GemmaSAM/
  README.md                         # This file
  requirements.txt                  # Python dependencies
  single_inference.ipynb            # Colab/Jupyter notebook demo

  infer/
    run_single_inference.py         # CLI entry point: parses args and runs inference
    unibiomed_inference_toolusing.py  # Core inference logic + Gemma grounding model

    models/
      __init__.py                   # (empty) package marker
      model_loader.py               # MedSAM2 segmentation model wrapper + model loading

    utils/
      __init__.py                   # (empty) package marker
      clicker.py                    # Click/point management for interactive segmentation
      visual_utils.py               # Visualization helpers (overlays, annotations)

    demo/
      Normal_abdominal_organs_CT_scan.png  # Demo CT image
      Brain.jpg                             # Demo brain MRI image
```

