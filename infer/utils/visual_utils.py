"""
Visualisation utilities for GemmaSAM segmentation results.

Provides functions to overlay masks, click points, and bounding boxes
on medical images, then save the composited result as JPEG/PNG files.

All functions handle both NumPy arrays and PyTorch tensors transparently.
"""

import os

import cv2
import numpy as np
import torch


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    """Overlay a semi-transparent coloured mask on an image.

    The mask is blended with the image::

        result = image * (1 - alpha) + colour * alpha   (where mask > 0)

    Args:
        image:  (H, W, 3) RGB numpy array.
        mask:   (H, W) binary mask (>0 = foreground).
        color:  RGB tuple for the overlay colour.
        alpha:  Opacity (0 = fully transparent, 1 = fully opaque).

    Returns:
        (H, W, 3) uint8 array with the mask overlaid.
    """
    overlay = image.copy()
    binary_mask = mask
    color_mask = np.zeros_like(image)
    color_mask[binary_mask > 0] = color

    for c in range(3):
        overlay[:, :, c] = np.where(
            binary_mask > 0,
            overlay[:, :, c] * (1 - alpha) + color_mask[:, :, c] * alpha,
            overlay[:, :, c],
        )
    return overlay


def overlay_points(image, points, color=(255, 0, 0)):
    """Draw a filled circle at a click point on the image.

    Args:
        image:  (H, W, 3) RGB numpy array.
        points: (y, x) coordinates (or a sequence).
        color:  BGR tuple (default red).

    Returns:
        Image with the circle drawn in-place.
    """
    if points is None:
        return image
    overlay = image.copy()
    cv2.circle(overlay, (int(points[1]), int(points[0])), 5, color, -1)
    return overlay


def overlay_boxes(image, box, color=(255, 0, 0), thickness=2):
    """Draw a bounding-box rectangle on the image.

    Args:
        image:     (H, W, 3) RGB numpy array.
        box:       [x1, y1, x2, y2] in pixel coordinates.
        color:     BGR tuple.
        thickness: Line thickness in pixels.

    Returns:
        Image with the rectangle drawn.
    """
    overlay = image.copy()
    overlay = cv2.rectangle(
        overlay,
        (int(box[0]), int(box[1])),
        (int(box[2]), int(box[3])),
        color,
        thickness,
    )
    return overlay


def visualize_mask_and_point(image, masks, points, path, is_positive,
                             text=None, box=None):
    """Render a single segmentation result with mask, point, optional box and text.

    This is the **single-click** visualisation — it draws one click point
    on the overlaid mask image and saves it to *path*.

    The point is colour-coded:
        - Red   for positive (foreground) clicks
        - Blue  for negative (background) clicks
        - Grey  when is_positive is None

    Args:
        image:       Source image (H, W, C) — NumPy or torch.Tensor.
        masks:       Predicted mask — NumPy or torch.Tensor.
        points:      (y, x) click coordinates (normalised or absolute).
        path:        Output file path (extension determines format).
        is_positive: True=positive, False=negative, None=unknown.
        text:        Optional caption text (auto-wrapped across lines).
        box:         Optional [x1, y1, x2, y2] bounding box.

    Returns:
        The rendered image as a (H, W, 3) numpy array.
    """
    # ── Normalise inputs ────────────────────────────────────────────────
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()    # C,H,W -> H,W,C
        print(image.min())
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
        if masks.ndim == 3:
            masks = masks[0]
    if points is not None:
        if points[0] < 1:                               # Normalised -> absolute
            points = [points[i] * image.shape[i] for i in range(2)]
        print(points)

    # ── Overlay mask ────────────────────────────────────────────────────
    overlay = overlay_mask(image, masks)

    # ── Overlay point (colour-coded) ────────────────────────────────────
    if points is not None:
        if is_positive is True:
            overlay = overlay_points(overlay, points)                     # red
        elif is_positive is False:
            overlay = overlay_points(overlay, points, color=(0, 0, 255))  # blue
        else:
            overlay = overlay_points(overlay, points, color=(128, 128, 128))  # grey

    # ── Overlay box ─────────────────────────────────────────────────────
    if box is not None:
        overlay = overlay_boxes(overlay, box)

    # ── Handle text caption ─────────────────────────────────────────────
    if text is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        font_thickness = 1
        max_width = overlay.shape[1] - 20

        # Word-wrap the text to fit the image width
        words = text.split(" ")
        lines = []
        current_line = words[0]
        for word in words[1:]:
            (w, _), _ = cv2.getTextSize(
                current_line + " " + word, font, font_scale, font_thickness
            )
            if w <= max_width:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        # Reserve space above the image for the text
        text_height = sum(
            cv2.getTextSize(line, font, font_scale, font_thickness)[0][1]
            for line in lines
        ) + 10 * len(lines)

        new_image = np.zeros(
            (overlay.shape[0] + text_height + 20, overlay.shape[1], 3),
            dtype=np.uint8,
        )
        new_image[text_height + 20:, :] = overlay

        y = 10
        for line in lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            text_x = (overlay.shape[1] - text_size[0]) // 2
            cv2.putText(
                new_image, line,
                (text_x, y + text_size[1]),
                font, font_scale, font_color, font_thickness,
            )
            y += text_size[1] + 10

        overlay = new_image

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(path), exist_ok=True)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, overlay_bgr)
    return overlay_bgr


def visualize_mask_and_pointlist(image, masks, point_list, path,
                                 is_positive_list, text=None, box=None):
    """Render a segmentation result with **multiple** click points, optional box and text.

    This is the **multi-click** variant used by the interactive loop to show
    all accumulated clicks from the VLM.  Each point is drawn individually:

        - Red   for positive clicks
        - Blue  for negative clicks

    Args:
        image:            Source image (H, W, C) — NumPy or torch.Tensor.
        masks:            Predicted mask.
        point_list:       List of (y, x) click coordinates.
        path:             Output file path.
        is_positive_list: List[bool] — True for foreground, False for background.
        text:             Optional caption text.
        box:              Optional [x1, y1, x2, y2] bounding box.

    Returns:
        The rendered image as a (H, W, 3) numpy array.
    """
    # ── Normalise inputs ────────────────────────────────────────────────
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()    # C,H,W -> H,W,C
        print(image.min())
        if image.max() <= 1:
            image = (image * 255).astype(np.uint8)
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
        if masks.ndim == 3:
            masks = masks[0]

    # ── Overlay mask ────────────────────────────────────────────────────
    overlay = overlay_mask(image, masks)

    # ── Overlay all points ──────────────────────────────────────────────
    for point_coords, is_positive in zip(point_list, is_positive_list):
        if is_positive:
            overlay = overlay_points(overlay, point_coords)                     # red
        else:
            overlay = overlay_points(overlay, point_coords, color=(0, 0, 255))  # blue

    # ── Overlay box ─────────────────────────────────────────────────────
    if box is not None:
        overlay = overlay_boxes(overlay, box)

    # ── Handle text caption (word-wrapped, same logic as single-click) ──
    if text is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        font_thickness = 1
        max_width = overlay.shape[1] - 20

        words = text.split(" ")
        lines = []
        current_line = words[0]
        for word in words[1:]:
            (w, _), _ = cv2.getTextSize(
                current_line + " " + word, font, font_scale, font_thickness,
            )
            if w <= max_width:
                current_line += " " + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        text_height = sum(
            cv2.getTextSize(line, font, font_scale, font_thickness)[0][1]
            for line in lines
        ) + 10 * len(lines)

        new_image = np.zeros(
            (overlay.shape[0] + text_height + 20, overlay.shape[1], 3),
            dtype=np.uint8,
        )
        new_image[text_height + 20:, :] = overlay

        y = 10
        for line in lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale, font_thickness)
            text_x = (overlay.shape[1] - text_size[0]) // 2
            cv2.putText(
                new_image, line,
                (text_x, y + text_size[1]),
                font, font_scale, font_color, font_thickness,
            )
            y += text_size[1] + 10

        overlay = new_image

    # ── Save ────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(path), exist_ok=True)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, overlay_bgr)
    return overlay_bgr
