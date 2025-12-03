import os
import numpy as np
import raft_dic_gui.processing as proc
import raft_dic_gui.model as mdl
from raft_dic_gui.config import DICConfig

class DICProcessor:
    """
    Controller class for RAFT-DIC processing.
    Handles the main processing loop, interacting with the processing module
    and reporting progress back to the GUI via callbacks.
    """
    def __init__(self, update_progress_callback=None, check_stop_callback=None):
        """
        Initialize the processor.
        :param update_progress_callback: Callable(percent, current, total) to update UI progress.
        :param check_stop_callback: Callable() -> bool that returns True if processing should stop.
        """
        self.update_progress = update_progress_callback
        self.check_stop = check_stop_callback
        self.displacement_results = []

    def run(self, config: DICConfig, roi_mask: np.ndarray, roi_rect: tuple):
        """
        Execute the processing pipeline.
        :param config: DICConfig object with all parameters.
        :param roi_mask: Boolean mask of the ROI.
        :param roi_rect: Tuple (xmin, ymin, xmax, ymax) of the ROI bounding box.
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
        
        # Get image file list
        image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'))
        ])

        if len(image_files) < 2:
            raise Exception("At least 2 images are needed")

        if roi_mask is None:
            raise Exception("ROI mask is missing")

        xmin, ymin, xmax, ymax = roi_rect

        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)
        
        # Smart Resume: Check if configuration has changed
        import json
        config_path = os.path.join(out_dir, "run_config.json")
        current_config_dict = {
            "model_path": config.model_path,
            "tile_overlap": config.tile_overlap,
            "context_padding": config.context_padding,
            "safety_factor": config.safety_factor,
            "use_smooth": config.use_smooth,
            "sigma": config.sigma,
            "p_max_pixels": config.p_max_pixels,
            "mode": config.mode,
            "roi_rect": list(roi_rect) # Add ROI to config check
        }
        
        force_rerun = False
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                
                # Compare critical parameters
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
            import glob
            npy_dir = os.path.join(out_dir, "displacement_results_npy")
            if os.path.exists(npy_dir):
                print(f"[INFO] Cleaning up stale results in {npy_dir}...")
                files = glob.glob(os.path.join(npy_dir, "displacement_field_*.npy"))
                for f in files:
                    try:
                        os.remove(f)
                    except Exception as e:
                        print(f"[WARNING] Failed to delete stale file {f}: {e}")

        # Load reference image
        ref_img_path = os.path.join(img_dir, image_files[0])
        ref_image = proc.load_and_convert_image(ref_img_path)
        ref_roi = ref_image[ymin:ymax, xmin:xmax]
        
        total_pairs = len(image_files) - 1
        
        if self.update_progress:
            self.update_progress(0, 0, total_pairs)
            
        sequence_displacements = []
        last_update_time = 0
        import time

        for i in range(1, len(image_files)):
            # Check for stop request
            if self.check_stop and self.check_stop():
                break
                
            # Check if result already exists (Resume functionality)
            result_filename = f"displacement_field_{i}.npy"
            result_path = os.path.join(out_dir, "displacement_results_npy", result_filename)
            
            # Load deformed image (needed for incremental mode or processing)
            def_img_path = os.path.join(img_dir, image_files[i])
            def_image = proc.load_and_convert_image(def_img_path)
            def_roi = def_image[ymin:ymax, xmin:xmax]

            if not force_rerun and os.path.exists(result_path):
                print(f"[INFO] Frame {i} already processed. Loading from {result_filename}...")
                try:
                    displacement_field = np.load(result_path)
                    sequence_displacements.append(displacement_field)
                    
                    # Update progress even if skipped
                    current_time = time.time()
                    if self.update_progress and (current_time - last_update_time > 0.1 or i == total_pairs):
                        percent = (i / max(1, total_pairs)) * 100.0
                        self.update_progress(percent, i, total_pairs)
                        last_update_time = current_time
                        
                    # If incremental mode, update reference image for NEXT frame
                    if config.mode == "incremental":
                        ref_roi = def_roi.copy()
                        ref_image = def_image.copy()
                        
                    continue # Skip processing
                except Exception as e:
                    print(f"[WARNING] Failed to load existing result for frame {i}: {e}. Reprocessing.")
            
            # Update progress (Rate limited to 10Hz)
            current_time = time.time()
            if self.update_progress and (current_time - last_update_time > 0.1 or i == total_pairs):
                percent = (i / max(1, total_pairs)) * 100.0
                self.update_progress(percent, i, total_pairs)
                last_update_time = current_time

            # Always use memory-safe tiled DIC over ROI
            disp_full, _ = proc.dic_over_roi_with_tiling(
                ref_image,
                def_image,
                roi_mask,
                model,
                device,
                context_padding=config.context_padding,
                tile_overlap=config.tile_overlap,
                p_max_pixels=config.p_max_pixels,
                use_smooth=config.use_smooth,
                sigma=config.sigma
            )
            displacement_field = disp_full[ymin:ymax, xmin:xmax, :]

            # Save immediately (Resume functionality)
            try:
                proc.save_displacement_results(
                    displacement_field,
                    out_dir,
                    index=i,
                    roi_rect=roi_rect,
                    roi_mask=roi_mask
                )
            except Exception as e:
                print(f"[ERROR] Failed to save result for frame {i}: {e}")

            # Accumulate this frame displacement
            sequence_displacements.append(displacement_field)

            # If incremental mode, update reference image(s)
            if config.mode == "incremental":
                ref_roi = def_roi.copy()
                ref_image = def_image.copy()

        # Save consolidated files
        try:
            proc.save_displacement_sequence(
                sequence_displacements,
                out_dir,
                roi_rect=roi_rect,
                roi_mask=roi_mask,
                save_numpy=True,
                save_matlab=True,
                numpy_filename="displacement_sequence.npz",
                matlab_filename="displacement_sequence.mat",
            )
        except Exception as e:
            print(f"Warning: Failed to save consolidated results: {e}")

        self.displacement_results = sequence_displacements
        return sequence_displacements
