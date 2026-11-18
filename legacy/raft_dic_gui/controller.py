import os
import json
import time
import numpy as np
import raft_dic_gui.processing as proc
import raft_dic_gui.model as mdl
from raft_dic_gui.config import DICConfig
from raft_dic_gui.incremental import (
    build_ref_map,
    key_frames_every_n,
    accumulate_displacement,
    median_filter_displacement,
)


class DICProcessor:
    """
    Controller class for RAFT-DIC processing.
    Handles the main processing loop, interacting with the processing module
    and reporting progress back to the GUI via callbacks.

    The processing loop is driven by a **ref_map** (frame -> reference frame,
    both 1-indexed).  Different strategies (accumulative, key-frame intervals,
    explicit key frames) simply produce different ref_maps.
    """

    def __init__(self, update_progress_callback=None, check_stop_callback=None):
        """
        Initialize the processor.
        :param update_progress_callback: Callable(percent, current, total, **extra) to update UI progress.
        :param check_stop_callback: Callable() -> bool that returns True if processing should stop.
        """
        self.update_progress = update_progress_callback
        self.check_stop = check_stop_callback
        self.displacement_results = []
        self._extra_progress = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, config: DICConfig, roi_mask: np.ndarray, roi_rect: tuple,
            image_files: list = None, user_masks: dict = None):
        """
        Execute the processing pipeline.
        :param config: DICConfig object with all parameters.
        :param roi_mask: Boolean mask of the ROI (full image size).
        :param roi_rect: Tuple (xmin, ymin, xmax, ymax) of the ROI bounding box.
        :param image_files: Pre-sorted list of image filenames.  When provided,
            the processor uses this order directly instead of re-scanning the
            directory (which would lose natural sort order).
        :param user_masks: Optional dict of 0-based frame index -> bool mask.
            When provided, these masks are used directly instead of
            discover_masks() from mask_dir.
        :return: List of displacement fields (or paths to them).
        """
        img_dir = config.img_dir
        out_dir = config.project_root

        # Load model
        device = config.device
        try:
            model = mdl.load_model(config.model_path, device=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {config.model_path}: {e}")

        # Use caller-provided file list (preserves natural sort) or fall back
        # to scanning the directory (for standalone / legacy usage).
        if image_files is None:
            import re as _re

            def _natural_sort_key(s: str):
                return [int(p) if p.isdigit() else p.lower()
                        for p in _re.split(r"(\d+)", s)]

            image_files = sorted(
                [f for f in os.listdir(img_dir)
                 if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'))],
                key=_natural_sort_key,
            )

        if len(image_files) < 2:
            raise Exception("At least 2 images are needed")

        # Store absolute paths for export
        self.image_files = [os.path.abspath(os.path.join(img_dir, f)) for f in image_files]

        if roi_mask is None:
            raise Exception("ROI mask is missing")

        xmin, ymin, xmax, ymax = roi_rect
        H_roi = ymax - ymin
        W_roi = xmax - xmin

        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)

        # ----------------------------------------------------------
        # Build ref_map
        # ----------------------------------------------------------
        total_frames = len(image_files)  # 1-indexed: frames 1..total_frames
        ref_map = self._build_ref_map(config, total_frames)

        # Set of frames that are referenced by other frames (for caching)
        key_frame_set = set(ref_map.values())
        key_frame_set.add(1)  # Frame 1 is always a key frame

        print(f"[INFO] ref_map built: {len(ref_map)} entries, "
              f"key frames = {sorted(key_frame_set)}")

        # ----------------------------------------------------------
        # Smart Resume: Check if configuration has changed
        # ----------------------------------------------------------
        force_rerun = self._check_config_changed(
            config, roi_rect, out_dir, image_files
        )

        # ----------------------------------------------------------
        # Discover per-frame user masks (if mask_dir configured)
        # ----------------------------------------------------------
        if user_masks is None:
            user_masks = {}
            if config.mask_dir:
                from raft_dic_gui.mask_loader import discover_masks
                ref_image_tmp = proc.load_and_convert_image(
                    os.path.join(img_dir, image_files[0])
                )
                image_shape = (ref_image_tmp.shape[0], ref_image_tmp.shape[1])
                user_masks = discover_masks(config.mask_dir, image_files, image_shape)
                del ref_image_tmp

        # ----------------------------------------------------------
        # State caches
        # ----------------------------------------------------------
        # kf_images[frame_num] = loaded full-size image for reference frames
        kf_images = {}

        # Load frame 1 image (always needed as a reference)
        ref1_path = os.path.join(img_dir, image_files[0])
        kf_images[1] = proc.load_and_convert_image(ref1_path)

        # kf_accumulated[frame_num] = total displacement (H_full, W_full, 2)
        # relative to Frame 1 for each key frame.
        # Full-image arrays: NaN outside ROI, zero inside.
        H_full, W_full = kf_images[1].shape[:2]
        kf_acc_init = np.full((H_full, W_full, 2), np.nan, dtype=np.float64)
        kf_acc_init[roi_mask, :] = 0.0
        kf_accumulated = {1: kf_acc_init}

        total_pairs = total_frames - 1

        if self.update_progress:
            self.update_progress(0, 0, total_pairs)

        sequence_displacements = []
        last_update_time = 0
        loop_start_time = time.time()

        # ----------------------------------------------------------
        # Main processing loop: frames 2..N  (1-indexed)
        # ----------------------------------------------------------
        for frame_num in range(2, total_frames + 1):
            # frame_num is 1-indexed; list index is frame_num - 1
            list_idx = frame_num - 1
            pair_idx = frame_num - 1  # pair index for progress (1-based pair count)

            # Check for stop request
            if self.check_stop and self.check_stop():
                break

            # ----------------------------------------------------------
            # Resume: try to load cached result
            # ----------------------------------------------------------
            result_filename = f"displacement_field_{list_idx}.npy"
            result_path = os.path.join(
                out_dir, "displacement_results_temp_npy", result_filename
            )

            if not force_rerun and os.path.exists(result_path):
                loaded = self._try_load_cached(
                    result_path, list_idx, frame_num, key_frame_set,
                    kf_accumulated, kf_images, img_dir, image_files,
                    H_full, W_full, roi_rect,
                )
                if loaded is not None:
                    sequence_displacements.append(loaded)
                    # Update progress even for cached frames
                    last_update_time = self._update_progress_throttled(
                        pair_idx, total_pairs, last_update_time
                    )
                    continue

            # ----------------------------------------------------------
            # Progress update (rate-limited to 10Hz)
            # ----------------------------------------------------------
            last_update_time = self._update_progress_throttled(
                pair_idx, total_pairs, last_update_time
            )

            # ----------------------------------------------------------
            # Determine reference and load images
            # ----------------------------------------------------------
            ref_num = ref_map[frame_num]

            # Ensure reference image is loaded
            if ref_num not in kf_images:
                ref_path = os.path.join(img_dir, image_files[ref_num - 1])
                kf_images[ref_num] = proc.load_and_convert_image(ref_path)

            ref_image = kf_images[ref_num]

            # Load deformed image
            def_img_path = os.path.join(img_dir, image_files[list_idx])
            def_image = proc.load_and_convert_image(def_img_path)

            # ----------------------------------------------------------
            # Determine mask for this frame
            # ----------------------------------------------------------
            current_mask = self._resolve_mask(
                frame_num, list_idx, ref_num,
                roi_mask, kf_accumulated, user_masks,
                xmin, ymin, xmax, ymax,
                auto_warp=(config.mask_dir is None),
            )

            # ----------------------------------------------------------
            # Update timing + VRAM in extra progress
            # ----------------------------------------------------------
            elapsed = time.time() - loop_start_time
            if pair_idx > 1:
                est_remaining = elapsed / (pair_idx - 1) * (total_pairs - pair_idx + 1)
            else:
                est_remaining = 0
            self._extra_progress.update({
                "elapsed_seconds": round(elapsed, 1),
                "est_remaining_seconds": round(est_remaining, 1),
            })
            try:
                import torch
                if torch.cuda.is_available():
                    free_mem, total_mem = torch.cuda.mem_get_info(0)
                    self._extra_progress["vram_used_mb"] = round((total_mem - free_mem) / 1024 / 1024)
                    self._extra_progress["vram_total_mb"] = round(total_mem / 1024 / 1024)
            except Exception:
                pass

            # ----------------------------------------------------------
            # Run DIC
            # ----------------------------------------------------------
            def _tile_cb(tile_idx, tile_total):
                self._extra_progress["tile_current"] = tile_idx + 1
                self._extra_progress["tile_total"] = tile_total

            disp_full, _ = proc.dic_over_roi_with_tiling(
                ref_image,
                def_image,
                current_mask,
                model,
                device,
                context_padding=config.context_padding,
                tile_overlap=config.tile_overlap,
                p_max_pixels=config.p_max_pixels,
                use_smooth=config.use_smooth,
                sigma=config.sigma,
                tile_callback=_tile_cb,
            )
            # Keep full-image for accumulation (disp_full already full-image
            # from dic_over_roi_with_tiling)
            delta_disp_full = disp_full

            # ----------------------------------------------------------
            # Apply current frame's mask: remove pixels whose deformed
            # position falls outside the specimen boundary (e.g. cracks)
            # ----------------------------------------------------------
            if list_idx in user_masks:
                self._apply_current_frame_mask(
                    delta_disp_full, user_masks[list_idx], frame_num,
                )

            # ----------------------------------------------------------
            # Compute total displacement relative to Frame 1
            # ----------------------------------------------------------
            kf_acc = kf_accumulated[ref_num]
            # Check if first segment: all valid pixels in accumulated are zero
            valid_acc = kf_acc[~np.isnan(kf_acc[..., 0])]
            is_first_segment = (len(valid_acc) == 0 or
                                np.allclose(valid_acc, 0, atol=1e-12))

            if is_first_segment:
                # First segment (ref == frame 1 with zero accumulated):
                # delta IS the total displacement
                total_disp_full = delta_disp_full.copy()
            else:
                # Subsequent segments: accumulate
                total_disp_full = accumulate_displacement(
                    kf_acc, delta_disp_full, debug=True
                )

            # ----------------------------------------------------------
            # Crop to roi_rect for storage (displacement is on Frame 1's grid)
            # ----------------------------------------------------------
            displacement_field = total_disp_full[ymin:ymax, xmin:xmax, :]

            # ----------------------------------------------------------
            # Optional median filter (reduces incremental error buildup)
            # ----------------------------------------------------------
            if not is_first_segment and config.use_median_filter:
                displacement_field = median_filter_displacement(displacement_field)

            # ----------------------------------------------------------
            # Cache key frame accumulated displacement
            # ----------------------------------------------------------
            if frame_num in key_frame_set:
                kf_accumulated[frame_num] = total_disp_full.copy()
                # Load and cache this frame's image for future reference
                if frame_num not in kf_images:
                    kf_images[frame_num] = def_image
                print(f"[INFO] Cached key frame {frame_num} accumulated displacement")

            # ----------------------------------------------------------
            # Evict reference images and accumulated state no longer needed
            # ----------------------------------------------------------
            self._evict_unused_ref_state(
                frame_num, ref_map, key_frame_set,
                kf_images, kf_accumulated, total_frames,
            )

            # ----------------------------------------------------------
            # Warn if significant pixel loss
            # ----------------------------------------------------------
            valid_pixels = int(np.sum(~np.isnan(displacement_field[..., 0])))
            total_pixels = H_roi * W_roi
            loss_pct = (1 - valid_pixels / total_pixels) * 100
            if loss_pct > 10:
                print(
                    f"[WARNING] Frame {frame_num}: {loss_pct:.1f}% of ROI pixels "
                    f"lost ({valid_pixels}/{total_pixels} valid)."
                )

            # ----------------------------------------------------------
            # Save immediately (resume support)
            # ----------------------------------------------------------
            try:
                proc.save_displacement_results(
                    displacement_field,
                    out_dir,
                    index=list_idx,
                    roi_rect=roi_rect,
                    roi_mask=roi_mask,
                )
            except Exception as e:
                print(f"[ERROR] Failed to save result for frame {frame_num}: {e}")

            sequence_displacements.append(displacement_field)

        self.displacement_results = sequence_displacements
        self.envelope_rect = roi_rect  # For now, same as roi_rect
        return sequence_displacements

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ref_map(config: DICConfig, total_frames: int):
        """Build the frame -> reference mapping from config settings.

        For accumulative mode, always use kf=[1] (all frames reference
        frame 1).  Incremental params (key_frame_interval, key_frames)
        are ignored to prevent config leak when switching modes.

        For incremental mode:
        1. key_frame_interval -> generate key frames via key_frames_every_n()
        2. key_frames -> use directly
        3. Default: every frame is a key frame
        """
        if config.mode != "incremental":
            # Accumulative: single segment, all frames ref frame 1
            return build_ref_map(total_frames, key_frames=[1])

        # Incremental mode
        kf = None
        if config.key_frame_interval is not None and config.key_frame_interval >= 1:
            kf = key_frames_every_n(total_frames, config.key_frame_interval)
        elif config.key_frames is not None and len(config.key_frames) > 0:
            kf = list(config.key_frames)

        if kf is None:
            # Classic incremental DIC: every frame is a key frame
            kf = list(range(1, total_frames + 1))

        return build_ref_map(total_frames, key_frames=kf)

    def _check_config_changed(self, config: DICConfig, roi_rect: tuple,
                              out_dir: str, image_files: list = None) -> bool:
        """Compare current config to saved config. Returns True if force rerun needed."""
        import hashlib
        config_path = os.path.join(out_dir, "run_config.json")

        # Serialize key_frames as sorted list for stable comparison
        kf_serialized = sorted(config.key_frames) if config.key_frames else None

        # Image set fingerprint: hash of sorted filenames + count
        # Detects image directory change, file additions/deletions
        if image_files:
            img_fingerprint = hashlib.sha256(
                "\n".join(image_files).encode()
            ).hexdigest()[:16]
        else:
            img_fingerprint = None

        current_config_dict = {
            "model_path": config.model_path,
            "img_dir": config.img_dir,
            "image_fingerprint": img_fingerprint,
            "tile_overlap": config.tile_overlap,
            "context_padding": config.context_padding,
            "use_smooth": config.use_smooth,
            "sigma": config.sigma,
            "p_max_pixels": config.p_max_pixels,
            "mode": config.mode,
            "roi_rect": list(roi_rect),
            # Incremental mode parameters
            "key_frames": kf_serialized,
            "key_frame_interval": config.key_frame_interval,
            "mask_dir": config.mask_dir,
            "use_median_filter": config.use_median_filter,
        }

        force_rerun = False
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                if saved_config != current_config_dict:
                    print("[INFO] Configuration changed. Forcing re-run.")
                    force_rerun = True
                else:
                    print("[INFO] Configuration matches. Resuming...")
            except Exception as e:
                print(f"[WARNING] Failed to read run config: {e}. Forcing re-run.")
                force_rerun = True

        # Save current config
        try:
            with open(config_path, 'w') as f:
                json.dump(current_config_dict, f, indent=4)
        except Exception as e:
            print(f"[WARNING] Failed to save run config: {e}")

        # Cleanup stale results if forcing re-run
        if force_rerun:
            import glob as glob_mod
            npy_dir = os.path.join(out_dir, "displacement_results_temp_npy")
            if os.path.exists(npy_dir):
                print(f"[INFO] Cleaning up stale results in {npy_dir}...")
                files = glob_mod.glob(os.path.join(npy_dir, "displacement_field_*.npy"))
                for f in files:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"[WARNING] Failed to delete stale file {f}: {e}")

        return force_rerun

    def _try_load_cached(
        self, result_path, list_idx, frame_num, key_frame_set,
        kf_accumulated, kf_images, img_dir, image_files,
        H_full=None, W_full=None, roi_rect=None,
    ):
        """Try to load a cached .npy result. Returns the array or None on failure.

        If the frame is a key frame, restores kf_accumulated and kf_images.
        Cached .npy files are crop-sized (H_roi, W_roi, 2), so they must be
        embedded into full-image arrays for kf_accumulated.
        """
        print(f"[INFO] Frame {frame_num} already processed. Loading from cache...")
        try:
            displacement_field = np.load(result_path)

            # If this frame is a key frame, restore accumulated state
            if frame_num in key_frame_set:
                # Cached result is crop-sized (H_roi, W_roi, 2).
                # Embed into full-image for kf_accumulated.
                if H_full is not None and W_full is not None and roi_rect is not None:
                    xmin, ymin, xmax, ymax = roi_rect
                    full = np.full((H_full, W_full, 2), np.nan, dtype=np.float64)
                    full[ymin:ymax, xmin:xmax, :] = displacement_field
                    kf_accumulated[frame_num] = full
                else:
                    # Fallback: use crop-sized directly (shouldn't happen)
                    kf_accumulated[frame_num] = displacement_field.copy()
                # Load the image for future reference if not already cached
                if frame_num not in kf_images:
                    img_path = os.path.join(img_dir, image_files[frame_num - 1])
                    kf_images[frame_num] = proc.load_and_convert_image(img_path)
                print(f"[INFO] Restored key frame {frame_num} state from cache")

            return displacement_field
        except Exception as e:
            print(
                f"[WARNING] Failed to load existing result for frame "
                f"{frame_num}: {e}. Reprocessing."
            )
            return None

    @staticmethod
    def _resolve_mask(
        frame_num, list_idx, ref_num,
        roi_mask, kf_accumulated, user_masks,
        xmin, ymin, xmax, ymax,
        auto_warp=False,
    ):
        """Determine the mask for the current frame.

        Fallback chain:
        1. Per-frame user mask (from discover_masks, keyed by 0-based index)
        2. Auto-warped ROI mask (if enabled and accumulated displacement available)
        3. Original ROI mask
        """
        # Priority 1: user-provided per-frame mask (keyed by reference frame,
        # 0-indexed).  Mask M_k is the ROI on image I_k and is used for the
        # pair I_k -> I_{k+1}.
        ref_idx = ref_num - 1
        if ref_idx in user_masks:
            print(f"[INFO] Frame {frame_num}: using user-provided mask "
                  f"(ref frame {ref_num})")
            return user_masks[ref_idx]

        # Priority 2: auto-warp using accumulated displacement
        if auto_warp and ref_num != 1 and ref_num in kf_accumulated:
            acc = kf_accumulated[ref_num]
            valid_count = np.sum(~np.isnan(acc[..., 0]))
            if valid_count > 0:
                from raft_dic_gui.incremental import warp_mask_with_holes
                warped = warp_mask_with_holes(roi_mask, acc)
                warped_count = int(np.sum(warped))
                orig_count = int(np.sum(roi_mask))
                print(f"[INFO] Frame {frame_num}: auto-warped mask "
                      f"({warped_count} px, {warped_count/max(1,orig_count)*100:.0f}% of original)")
                return warped

        # Priority 3: original ROI mask
        return roi_mask

    @staticmethod
    def _apply_current_frame_mask(disp_full, cur_mask, frame_num):
        """Mask out pixels whose deformed position falls outside cur_mask.

        For each valid pixel (x, y) with displacement (U, V), check whether
        cur_mask at the deformed position (round(y+V), round(x+U)) is True.
        Pixels landing inside the image but outside the specimen (e.g. in a
        crack, or inside a bubble hole) are set to NaN.

        Pixels whose deformed position falls OUTSIDE the image bounds are
        preserved — they represent material physically pushed out of view
        (e.g. large bubble expansion, large strain near edges).  Discarding
        them would punch holes in the reference-frame Delaunay coverage used
        by deformed-view inverse mapping, producing a 5-10 px dead band around
        the deformed image boundary.
        """
        H, W = disp_full.shape[:2]
        valid = ~np.isnan(disp_full[..., 0])
        yy, xx = np.where(valid)

        if len(yy) == 0:
            return

        U = disp_full[yy, xx, 0]
        V = disp_full[yy, xx, 1]

        # Deformed coordinates (where these reference pixels land)
        def_x = np.round(xx + U).astype(np.intp)
        def_y = np.round(yy + V).astype(np.intp)

        in_bounds = (def_x >= 0) & (def_x < W) & (def_y >= 0) & (def_y < H)

        # Default: pixel is kept (True).  OOB pixels stay True because their
        # forward-mapped position being off-image is physical, not a defect.
        in_specimen = np.ones(len(yy), dtype=bool)
        in_specimen[in_bounds] = cur_mask[def_y[in_bounds], def_x[in_bounds]]

        outside = ~in_specimen
        if outside.any():
            disp_full[yy[outside], xx[outside], :] = np.nan
            print(f"[INFO] Frame {frame_num}: current-frame mask removed "
                  f"{outside.sum()} pixels outside specimen boundary")

    def _update_progress_throttled(self, pair_idx, total_pairs, last_update_time):
        """Update progress callback, rate-limited to ~10Hz. Returns new last_update_time."""
        current_time = time.time()
        if self.update_progress and (
            current_time - last_update_time > 0.1 or pair_idx == total_pairs
        ):
            percent = (pair_idx / max(1, total_pairs)) * 100.0
            self.update_progress(percent, pair_idx, total_pairs, **self._extra_progress)
            return current_time
        return last_update_time

    @staticmethod
    def _evict_unused_ref_state(
        frame_num, ref_map, key_frame_set,
        kf_images, kf_accumulated, total_frames,
    ):
        """Remove cached reference images and accumulated displacements
        that are no longer needed by future frames.

        A key frame's state is needed only while there are future frames
        that reference it.  Once we've processed past the last such frame,
        both the image and the accumulated displacement can be evicted.
        """
        # Check each cached key frame (except frame 1 which is always needed)
        to_evict = []
        all_cached = set(kf_images.keys()) | set(kf_accumulated.keys())
        for cached_kf in all_cached:
            if cached_kf == 1:
                continue
            still_needed = False
            for future_frame in range(frame_num + 1, total_frames + 1):
                if future_frame in ref_map and ref_map[future_frame] == cached_kf:
                    still_needed = True
                    break
            if not still_needed:
                to_evict.append(cached_kf)

        for kf in to_evict:
            evicted = []
            if kf in kf_images:
                del kf_images[kf]
                evicted.append("image")
            if kf in kf_accumulated:
                del kf_accumulated[kf]
                evicted.append("accumulated")
            if evicted:
                print(f"[INFO] Evicted {', '.join(evicted)} for key frame {kf}")
