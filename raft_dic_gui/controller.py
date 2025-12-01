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

        # Load reference image
        ref_img_path = os.path.join(img_dir, image_files[0])
        ref_image = proc.load_and_convert_image(ref_img_path)
        
        # Extract ROI area
        ref_roi = ref_image[ymin:ymax, xmin:xmax]

        # Determine processing canvas
        H_ref, W_ref = ref_image.shape[:2]
        
        # Process each image pair
        total_pairs = len(image_files) - 1
        
        if self.update_progress:
            self.update_progress(0, 0, total_pairs)
            
        sequence_displacements = []

        for i in range(1, len(image_files)):
            # Check for stop request
            if self.check_stop and self.check_stop():
                break
                
            # Update progress
            if self.update_progress:
                percent = (i / max(1, total_pairs)) * 100.0
                self.update_progress(percent, i, total_pairs)

            # Load deformed image
            def_img_path = os.path.join(img_dir, image_files[i])
            def_image = proc.load_and_convert_image(def_img_path)
            def_roi = def_image[ymin:ymax, xmin:xmax]

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
