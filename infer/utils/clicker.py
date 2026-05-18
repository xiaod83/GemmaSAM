"""
Click management for interactive medical-image segmentation.

Provides data structures and logic for accumulating, undoing, and
automatically generating click points used by interactive segmentation
models (e.g., SAM2, MedSAM2).

Key classes:
  - Click:     A single positive (foreground) or negative (background) click.
  - Clicker:   Accumulates clicks with undo-radius support and automatic
               next-click selection via distance transforms.
  - Clicker_sampler:  Training-data variant that adds random click sampling.
"""

import numpy as np
from copy import deepcopy
import cv2
import matplotlib.pyplot as plt
import os


def ensure_intermediate_results_dir(dataset_name='default',
                                    output_dir='./intermediate_results') -> str:
    """Create and return the output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


class Clicker(object):
    """Manages a sequence of positive/negative clicks for interactive segmentation.

    Key features:
      - Accumulates clicks added via ``add_click()``.
      - Optional **undo radius**: if a new click has the opposite sign and is
        within ``radius`` pixels of an existing click, the old click is
        removed (useful for correcting accidental clicks).
      - ``make_next_click()`` uses the ground-truth mask to automatically
        determine the *next best click point* based on the largest error
        region (false negative / false positive) via distance transforms.
    """

    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1,
                 click_indx_offset=0, dataset_name='default',
                 output_dir='./intermediate_results'):
        """
        Args:
            gt_mask:            Ground-truth binary mask (for auto-click generation).
            init_clicks:        Optional list of Click objects to start with.
            ignore_label:       Label to ignore in the GT mask.
            click_indx_offset:  Starting index for click numbering.
            dataset_name:       Name used for visualization subdirectories.
            output_dir:         Base directory for saved visualizations.
        """
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None

        self.reset_clicks()
        self.visualize = False
        self.dataset_name = dataset_name
        self.visualize_dir = ensure_intermediate_results_dir(dataset_name, output_dir)
        self.object_id = "object"
        self.file_name = "unknown"

        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    # ──────────────────────────────────────────────────────────────────────
    #  Auto-click: determine the next click based on prediction errors
    # ──────────────────────────────────────────────────────────────────────

    def make_next_click(self, pred_mask, file_name=None):
        """Automatically determine the next click based on the current prediction.

        Requires ``self.gt_mask`` to be set.  Uses the largest-error
        distance-transform heuristic from RITM (Revising Interactive
        Segmentation with Transformers).

        Args:
            pred_mask:  Current predicted binary mask.
            file_name:  Name for saving visualizations.
        """
        assert self.gt_mask is not None
        self.file_name = file_name
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    # ──────────────────────────────────────────────────────────────────────
    #  Query clicks
    # ──────────────────────────────────────────────────────────────────────

    def get_clicks(self, clicks_limit=None):
        """Return all accumulated clicks, optionally limited to the first N."""
        return self.clicks_list[:clicks_limit]

    def get_last_click(self):
        """Return only the most recently added click (for iterative refinement)."""
        if len(self.clicks_list) > 0:
            return [self.clicks_list[-1]]
        return []

    # ──────────────────────────────────────────────────────────────────────
    #  Core next-click algorithm
    # ──────────────────────────────────────────────────────────────────────

    def _get_next_click(self, pred_mask, padding=True, save_dir='./visualizations'):
        """Determine the next best click location using distance transforms.

        The algorithm:
        1. Compute the false-negative (FN) and false-positive (FP) regions.
        2. Compute the distance transform of each region.
        3. Filter out already-clicked pixels (``not_clicked_map``).
        4. Pick the pixel with the largest distance — positive if FN > FP,
           negative otherwise.

        Args:
            pred_mask:  Current binary prediction.
            padding:    Add 1-pixel border padding before distance transform
                        to avoid edge artifacts.
            save_dir:   Directory for optional debug visualizations.

        Returns:
            A ``Click`` object with the next optimal point.
        """
        visualize = self.visualize
        if self.visualize:
            save_dir = self.visualize_dir
        if visualize and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        click_idx = len(self.clicks_list) + 1

        # False negative: GT says foreground, model says background
        fn_mask = np.logical_and(
            np.logical_and(self.gt_mask, np.logical_not(pred_mask)),
            self.not_ignore_mask,
        )
        # False positive: GT says background, model says foreground
        fp_mask = np.logical_and(
            np.logical_and(np.logical_not(self.gt_mask), pred_mask),
            self.not_ignore_mask,
        )

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("False Negative Mask (fn_mask)")
            plt.imshow(fn_mask, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title("False Positive Mask (fp_mask)")
            plt.imshow(fp_mask, cmap='gray')
            plt.savefig(os.path.join(
                save_dir,
                f'{self.file_name}_{str(self.object_id)}_fn_fp_masks_{click_idx}.png',
            ))
            plt.close()

        # Pad before distance transform to avoid edge zeroing
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_dt = cv2.distanceTransform(
            fn_mask.astype(np.uint8), cv2.DIST_L2, 0
        )
        fp_mask_dt = cv2.distanceTransform(
            fp_mask.astype(np.uint8), cv2.DIST_L2, 0
        )

        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Distance Transform of False Negative (fn_mask_dt)")
            plt.imshow(fn_mask_dt, cmap='jet')
            plt.subplot(1, 2, 2)
            plt.title("Distance Transform of False Positive (fp_mask_dt)")
            plt.imshow(fp_mask_dt, cmap='jet')
            plt.savefig(
                f'{save_dir}/{self.file_name}_{str(self.object_id)}_'
                f'distance_transforms_{click_idx}.png'
            )
            plt.close()

        # Mask out pixels that have already been clicked
        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map

        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("False Negative Distance Transform (Filtered)")
            plt.subplot(1, 2, 2)
            plt.title("False Positive Distance Transform (Filtered)")
            plt.savefig(
                f'{save_dir}/{self.file_name}_{str(self.object_id)}_'
                f'filtered_distance_transforms_{click_idx}.png'
            )
            plt.close()

        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)

        # Choose the type of click based on which error is larger
        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)

        if visualize:
            plt.figure(figsize=(5, 5))
            plt.title("Next Click Position")
            plt.imshow(pred_mask, cmap='hot')
            color = 'red' if is_positive else 'blue'
            plt.scatter(coords_x[0], coords_y[0], color=color, s=100)
            plt.savefig(os.path.join(
                save_dir,
                f'{self.file_name}_{str(self.object_id)}_'
                f'next_click_position_{click_idx}.png',
            ))
            plt.close()

        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    # ──────────────────────────────────────────────────────────────────────
    #  Click mutation
    # ──────────────────────────────────────────────────────────────────────

    def add_click(self, click, radius=0):
        """Add a click, optionally removing opposite-sign clicks within *radius*.

        The undo-radius logic: if a new click has the opposite sign and is
        within ``radius`` pixels of an existing click, the existing click is
        removed.  This allows the VLM or user to "undo" a previous click
        by clicking nearby with the opposite label.

        Args:
            click:  A Click object.
            radius: Undo radius in pixels (0 = disabled).
        """
        coords = click.coords
        if radius > 0:
            x1, y1 = click.coords
            p1 = click.is_positive
            for prev_click in list(self.clicks_list):
                x2, y2 = prev_click.coords
                p2 = prev_click.is_positive
                dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if dist < radius and p1 != p2:
                    self.clicks_list.remove(prev_click)
                    if prev_click.is_positive:
                        self.num_pos_clicks -= 1
                    else:
                        self.num_neg_clicks -= 1

        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def _remove_last_click(self):
        """Remove the most recently added click (undo helper)."""
        click = self.clicks_list.pop()
        coords = click.coords
        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        """Clear all accumulated clicks."""
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)
        self.num_pos_clicks = 0
        self.num_neg_clicks = 0
        self.clicks_list = []

    # ──────────────────────────────────────────────────────────────────────
    #  State serialisation
    # ──────────────────────────────────────────────────────────────────────

    def get_state(self):
        """Return a deep copy of the current click list (for saving/restoring)."""
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        """Restore a previously saved click state."""
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Clicker_sampler(object):
    """Variant of Clicker for sampling training data.

    Adds a ``random_sample_click()`` method that samples random positive
    and negative clicks from the ground-truth mask, useful for generating
    training examples for click-based segmentation models.
    """

    def __init__(self, gt_mask=None, init_clicks=None, ignore_label=-1,
                 click_indx_offset=0):
        self.click_indx_offset = click_indx_offset
        if gt_mask is not None:
            self.gt_mask = gt_mask == 1
            self.not_ignore_mask = gt_mask != ignore_label
        else:
            self.gt_mask = None
        self.reset_clicks()
        self.visualize = False
        if init_clicks is not None:
            for click in init_clicks:
                self.add_click(click)

    def make_next_click(self, pred_mask, file_name=None):
        """Same distance-transform-based auto-click as Clicker."""
        assert self.gt_mask is not None
        self.file_name = file_name
        click = self._get_next_click(pred_mask)
        self.add_click(click)

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def _get_next_click(self, pred_mask, padding=True, save_dir='./visualizations'):
        """Identical to Clicker._get_next_click (see that method for docs)."""
        visualize = self.visualize
        if self.visualize:
            save_dir = self.visualize_dir
        if visualize and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        click_idx = len(self.clicks_list) + 1
        fn_mask = np.logical_and(
            np.logical_and(self.gt_mask, np.logical_not(pred_mask)),
            self.not_ignore_mask,
        )
        fp_mask = np.logical_and(
            np.logical_and(np.logical_not(self.gt_mask), pred_mask),
            self.not_ignore_mask,
        )
        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("False Negative Mask (fn_mask)")
            plt.imshow(fn_mask, cmap='gray')
            plt.subplot(1, 2, 2)
            plt.title("False Positive Mask (fp_mask)")
            plt.imshow(fp_mask, cmap='gray')
            plt.savefig(os.path.join(
                save_dir,
                f'{self.file_name}_{str(self.object_id)}_fn_fp_masks_{click_idx}.png',
            ))
            plt.close()
        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')
        fn_mask_dt = cv2.distanceTransform(
            fn_mask.astype(np.uint8), cv2.DIST_L2, 0
        )
        fp_mask_dt = cv2.distanceTransform(
            fp_mask.astype(np.uint8), cv2.DIST_L2, 0
        )
        if padding:
            fn_mask_dt = fn_mask_dt[1:-1, 1:-1]
            fp_mask_dt = fp_mask_dt[1:-1, 1:-1]
        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Distance Transform of False Negative (fn_mask_dt)")
            plt.imshow(fn_mask_dt, cmap='jet')
            plt.subplot(1, 2, 2)
            plt.title("Distance Transform of False Positive (fp_mask_dt)")
            plt.imshow(fp_mask_dt, cmap='jet')
            plt.savefig(
                f'{save_dir}/{self.file_name}_{str(self.object_id)}_'
                f'distance_transforms_{click_idx}.png'
            )
            plt.close()
        fn_mask_dt = fn_mask_dt * self.not_clicked_map
        fp_mask_dt = fp_mask_dt * self.not_clicked_map
        if visualize:
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("False Negative Distance Transform (Filtered)")
            plt.subplot(1, 2, 2)
            plt.title("False Positive Distance Transform (Filtered)")
            plt.savefig(
                f'{save_dir}/{self.file_name}_{str(self.object_id)}_'
                f'filtered_distance_transforms_{click_idx}.png'
            )
            plt.close()
        fn_max_dist = np.max(fn_mask_dt)
        fp_max_dist = np.max(fp_mask_dt)
        is_positive = fn_max_dist > fp_max_dist
        if is_positive:
            coords_y, coords_x = np.where(fn_mask_dt == fn_max_dist)
        else:
            coords_y, coords_x = np.where(fp_mask_dt == fp_max_dist)
        if visualize:
            plt.figure(figsize=(5, 5))
            plt.title("Next Click Position")
            plt.imshow(pred_mask, cmap='hot')
            color = 'red' if is_positive else 'blue'
            plt.scatter(coords_x[0], coords_y[0], color=color, s=100)
            plt.savefig(os.path.join(
                save_dir,
                f'{self.file_name}_{str(self.object_id)}_'
                f'next_click_position_{click_idx}.png',
            ))
            plt.close()
        return Click(is_positive=is_positive, coords=(coords_y[0], coords_x[0]))

    def add_click(self, click):
        """Add a click (no undo-radius logic in the sampler variant)."""
        coords = click.coords
        click.indx = self.click_indx_offset + self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1
        self.clicks_list.append(click)
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = False

    def random_sample_click(self, pos_num: int, neg_num: int) -> None:
        """Sample *pos_num* random positive clicks and *neg_num* random negative clicks.

        Positive clicks are sampled from the foreground region of ``gt_mask``;
        negative clicks from the background.

        This is used during training data generation to simulate user clicks
        without requiring an interactive VLM loop.
        """
        pos_clicks, neg_clicks = [], []

        # Sample positive clicks from the GT foreground
        pos_coords = np.where(self.gt_mask)
        for _ in range(pos_num):
            idx = np.random.randint(len(pos_coords[0]))
            pos_clicks.append(
                Click(is_positive=True, coords=(pos_coords[0][idx], pos_coords[1][idx]))
            )

        # Sample negative clicks from the GT background
        neg_coords = np.where(np.logical_not(self.gt_mask))
        for _ in range(neg_num):
            idx = np.random.randint(len(neg_coords[0]))
            neg_clicks.append(
                Click(is_positive=False, coords=(neg_coords[0][idx], neg_coords[1][idx]))
            )

        for click in pos_clicks:
            self.add_click(click)
        for click in neg_clicks:
            self.add_click(click)

    def _remove_last_click(self):
        click = self.clicks_list.pop()
        coords = click.coords
        if click.is_positive:
            self.num_pos_clicks -= 1
        else:
            self.num_neg_clicks -= 1
        if self.gt_mask is not None:
            self.not_clicked_map[coords[0], coords[1]] = True

    def reset_clicks(self):
        if self.gt_mask is not None:
            self.not_clicked_map = np.ones_like(self.gt_mask, dtype=bool)
        self.num_pos_clicks = 0
        self.num_neg_clicks = 0
        self.clicks_list = []

    def get_state(self):
        return deepcopy(self.clicks_list)

    def set_state(self, state):
        self.reset_clicks()
        for click in state:
            self.add_click(click)

    def __len__(self):
        return len(self.clicks_list)


class Click:
    """A single interactive segmentation click.

    Attributes:
        is_positive (bool):  True for a foreground click, False for background.
        coords (tuple):      (y, x) pixel coordinates.
        indx (int):          Sequential index for ordering.
    """

    def __init__(self, is_positive: bool, coords: tuple, indx: int = None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        """Return (y, x, indx) — the format expected by SAM2's point encoding."""
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        """Return a deep copy, optionally overriding attributes via kwargs."""
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy
