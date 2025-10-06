"""
RAFT-DIC Helper Functions
------------------------
This module provides helper functions for the RAFT-DIC displacement field calculator.
It includes functions for:
- Image loading and preprocessing
- RAFT model loading and inference
- Displacement field calculation and visualization
- Result saving and data processing

Author: Zixiang (Zach) Tong @ UT-Austin, Lehu Bu @ UT-Austin
Date: 2025-06-03
Version: 1.0

Dependencies:
- OpenCV
- PyTorch
- Pillow
- NumPy
- SciPy
- Matplotlib
- tifffile
- RAFT core modules
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

import sys
import cv2
import time
import torch
import numpy as np
# import requests
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections as collections
import matplotlib.cm as cm
import warnings
import scipy.io as sio
import tifffile
from scipy.io import savemat
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.sparse import csc_matrix, eye
from scipy.sparse.linalg import spsolve

# Ignore specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Set matplotlib to use Agg backend to avoid font issues
plt.switch_backend('Agg')

sys.path.append('./core')  # Add raft folder to path
from core.raft import RAFT
from core.utils import flow_viz
from core.utils.utils import InputPadder

# A sketchy class to pass to RAFT (adjust parameters as needed)
class Args():
    def __init__(self, model='', path='', small=False, mixed_precision=True, alternate_corr=False):
        self.model = model
        self.path = path
        self.small = small
        self.mixed_precision = mixed_precision
        self.alternate_corr = alternate_corr
    def __iter__(self):
        return self
    def __next__(self):
        raise StopIteration
    
def process_img(img, device):
    """Converts a numpy image (H, W, 3) to a torch tensor of shape [1, 3, H, W]."""
    return torch.from_numpy(img).permute(2, 0, 1).float()[None].to(device)

def load_model(weights_path, args, weights_only=True):
    """Loads the RAFT model with given weights and arguments."""
    model = RAFT(args)
    pretrained_weights = torch.load(weights_path, map_location=torch.device("cpu"))
    model = torch.nn.DataParallel(model)
    model.load_state_dict(pretrained_weights)
    model.to("cuda")
    return model

def inference(model, frame1, frame2, device, pad_mode='sintel',
              iters=12, flow_init=None, upsample=True, test_mode=True):
    """Runs RAFT inference on a pair of images."""
    model.eval()
    with torch.no_grad():
        frame1 = process_img(frame1, device)
        frame2 = process_img(frame2, device)
        
        # Record original size
        original_size = (frame1.shape[2], frame1.shape[3])
        
        # Use padder for padding
        padder = InputPadder(frame1.shape, mode=pad_mode)
        frame1, frame2 = padder.pad(frame1, frame2)
        
        # Use new autocast syntax
        with torch.amp.autocast('cuda', enabled=True):
            if test_mode:
                flow_low, flow_up = model(frame1, frame2, iters=iters, 
                                        flow_init=flow_init,
                                        upsample=upsample, 
                                        test_mode=test_mode)
                
                # Crop back to original size
                flow_up = flow_up[:, :, :original_size[0], :original_size[1]]
                flow_low = flow_low[:, :, :original_size[0]//8, :original_size[1]//8]
                
                return flow_low, flow_up
            else:
                flow_iters = model(frame1, frame2, iters=iters, 
                                 flow_init=flow_init,
                                 upsample=upsample, 
                                 test_mode=test_mode)
                return flow_iters

def get_viz(flo):
    """Converts flow to a visualization image."""
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    return flow_viz.flow_to_image(flo)

def load_and_convert_image(img_path):
    """Load image from local path and convert color space and bit depth
    
    Args:
        img_path: Image file path
        
    Returns:
        frame_rgb: RGB format 8bit image
    """
    # Check file extension
    ext = os.path.splitext(img_path)[1].lower()
    
    if ext in ['.tif', '.tiff']:
        try:
            # Use tifffile to read TIFF files
            frame = tifffile.imread(img_path)
        except Exception as e:
            raise Exception(f"Failed to load TIFF image from {img_path}: {str(e)}")
    else:
        # Use OpenCV for other formats
        frame = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    if frame is None:
        raise Exception(f"Failed to load image from {img_path}")

    # Check image bit depth and type and convert accordingly
    if frame.dtype == np.float32:
        # For 32-bit float images, first normalize to 0-1 range
        min_val = np.nanmin(frame)  # Use nanmin to handle potential NaN values
        max_val = np.nanmax(frame)
        if max_val != min_val:
            frame = np.clip((frame - min_val) / (max_val - min_val), 0, 1)
        else:
            frame = np.zeros_like(frame)
        
        # Convert to 0-255 range and 8-bit unsigned integer
        frame = (frame * 255).astype(np.uint8)
    elif frame.dtype == np.uint16:
        # 16bit to 8bit conversion, preserving relative brightness
        frame = (frame / 256).astype(np.uint8)
    elif frame.dtype != np.uint8:
        # Other bit depths, normalize to 0-255 range
        frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)

    # If single channel image, convert to three channels
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] > 3:
        # If more than 3 channels, keep only first 3
        frame = frame[:, :, :3]
    
    # If image was read by tifffile, handle color channel order
    if ext in ['.tif', '.tiff']:
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # BGR to RGB conversion
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame_rgb

# ============================
# Cutting and Flow Reconstruction with Fixed-Stride Cropping
# ============================
def calculate_window_positions(image_size, crop_size, shift):
    """
    Calculate window positions to ensure complete coverage with full-size windows
    
    Args:
        image_size: Image dimensions
        crop_size: Crop window size
        shift: Sliding step size
    
    Returns:
        positions: List of window start positions
    """
    positions = []
    current = 0
    
    while current <= image_size - crop_size:
        positions.append(current)
        current += shift
    
    # If last position doesn't cover edge, add one that ensures edge coverage
    if current < image_size - crop_size:
        positions.append(image_size - crop_size)
    elif positions[-1] + crop_size < image_size:
        positions.append(image_size - crop_size)
    
    return positions


def cut_image_pair_with_flow(ref_img, def_img, project_root, model, device, 
                           crop_size=(128, 128), shift=64, maxDisplacement=50,
                           plot_windows=False, roi_mask=None,
                           use_smooth=True, sigma=2.0):
    """
    Process image pair and calculate displacement field
    
    Args:
        ref_img: Reference image
        def_img: Deformed image
        project_root: Project root directory
        model: RAFT model
        device: Computation device
        crop_size: Cutting window size
        shift: Sliding step size
        maxDisplacement: Maximum displacement
        plot_windows: Whether to plot window layout
        roi_mask: ROI mask, binary array same size as input image
        use_smooth: Whether to use smoothing
        sigma: Gaussian smoothing sigma parameter
    """
    start_total = time.time()  # Start timing total process
    
    # Create necessary subdirectories
    windows_dir = os.path.join(project_root, "windows")
    os.makedirs(windows_dir, exist_ok=True)

    H, W, _ = ref_img.shape
    crop_h, crop_w = crop_size

    # Calculate window positions in x and y directions
    x_positions = calculate_window_positions(W, crop_w, shift)
    y_positions = calculate_window_positions(H, crop_h, shift)

    windows = []
    global_flow = np.zeros((H, W, 2), dtype=np.float64)
    count_field = np.zeros((H, W), dtype=np.float64)
    # Create dictionary to store flow results for each window
    window_flows = {}
    
    # If ROI mask provided, ensure it's boolean type
    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)
    else:
        # If no mask provided, create all True mask
        roi_mask = np.ones((H, W), dtype=bool)

    # Valid mask (maintain original valid area calculation)
    confidenceRange_y = [maxDisplacement, crop_h-maxDisplacement]
    confidenceRange_x = [maxDisplacement, crop_w-maxDisplacement]
    valid_mask = np.zeros((crop_h, crop_w), dtype=np.float64)
    valid_mask[confidenceRange_y[0]:confidenceRange_y[1], 
              confidenceRange_x[0]:confidenceRange_x[1]] = 1.0

    count = 0
    inference_time = 0
    for y in y_positions:
        for x in x_positions:
            # Now all windows are full crop_size
            ref_window = ref_img[y:y+crop_h, x:x+crop_w, :]
            def_window = def_img[y:y+crop_h, x:x+crop_w, :]
            
            window_key = f"{x}_{y}"  # Use coordinates as key
            windows.append({
                'index': count,
                'position': (x, y, x+crop_w, y+crop_h),
                'key': window_key
            })

            # Add RAFT inference time statistics
            start_inference = time.time()
            flow_low, flow_up = inference(model, ref_window, def_window, device, test_mode=True)
            flow_up = flow_up.squeeze(0)
            inference_time += time.time() - start_inference
            
            u = flow_up[0].cpu().numpy()
            v = flow_up[1].cpu().numpy()
            window_flow = np.stack((u, v), axis=-1)

            # Save flow results for this window
            window_flows[window_key] = {
                'flow': window_flow * valid_mask[..., None],  # Apply valid_mask
                'position': (x, y, x+crop_w, y+crop_h),
                'index': count
            }

            global_flow[y:y+crop_h, x:x+crop_w, :] += window_flow * valid_mask[..., None]
            count_field[y:y+crop_h, x:x+crop_w] += valid_mask

            count += 1

    if False: # Save window_flows after loop
        ref_path = 'C:/Users/zt3323/OneDrive - The University of Texas at Austin/Documents/Python Codes/RAFT-2D-DIC-GUI/Results_window_256_step128_without_merging'
        save_dir = os.path.join(os.path.dirname(os.path.dirname(ref_path)), 'window_flows')
        os.makedirs(save_dir, exist_ok=True)
        
        # Use input image filename as base name for saving
        save_path = os.path.join(save_dir, f'{ref_path}/window_flows.mat')
        
        # Convert window_flows to format suitable for saving
        save_dict = {}
        for key, value in window_flows.items():
            save_dict[f'window_{key}'] = {
                'flow': value['flow'],
                'position': np.array(value['position']),
                'index': value['index']
            }
        # Save as mat file
        savemat(save_path, save_dict)


    # Calculate average displacement field
    final_flow = np.where(count_field[..., None] > 0,
                         global_flow / count_field[..., None],
                         np.nan)
    
    # Apply boolean mask
    final_flow[~roi_mask] = np.nan

    # Apply smoothing after displacement field calculation
    if use_smooth:
        displacement_field = smooth_displacement_field(final_flow, sigma=sigma)
    else:
        displacement_field = final_flow
        
    print(f"Total window pairs processed: {count}")

    # Save window layout plot
    if plot_windows:
        try:
            # Use Agg backend for thread safety
            plt.switch_backend('Agg')
            
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_RGB2GRAY)
            fig, ax = plt.subplots(1, figsize=(8, 8))
            ax.imshow(ref_gray, cmap='gray')
            ax.axis('off')
            colormap = cm.get_cmap('hsv', len(windows))
            
            # Create all rectangles at once instead of adding one by one
            patches_list = []
            for window in windows:
                x_start, y_start, x_end, y_end = window['position']
                w = x_end - x_start
                h = y_end - y_start
                color = colormap(window['index'])
                rect = patches.Rectangle((x_start, y_start), w, h, linewidth=2,
                                      edgecolor=color, facecolor='none')
                patches_list.append(rect)
            
            # Add all rectangles in batch
            ax.add_collection(collections.PatchCollection(patches_list, match_original=True))
            
            plt.title("Sliding Windows Layout")
            # Use bbox_inches='tight' to ensure complete image saving
            plt.savefig(os.path.join(windows_dir, "windows_layout.png"), 
                       bbox_inches='tight', dpi=300)
            plt.close(fig)  # Ensure figure is closed
            
        except Exception as e:
            print(f"Warning: Failed to save windows layout: {str(e)}")
            # Continue execution, don't let plotting error affect main processing
    
    total_time = time.time() - start_total
    
    print(f"\nTime statistics:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"RAFT inference time: {inference_time:.2f} seconds")
    print(f"RAFT inference percentage: {(inference_time/total_time*100):.1f}%")
    print(f"Other operations time: {(total_time-inference_time):.2f} seconds")
    
    return displacement_field, windows

def save_displacement_results(displacement_field, output_dir, index, roi_rect=None, roi_mask=None):
    """Save displacement field results in npy and mat formats with coordinate information"""
    # Create directory structure
    npy_dir = os.path.join(output_dir, "displacement_results_npy")
    mat_dir = os.path.join(output_dir, "displacement_results_mat")
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(mat_dir, exist_ok=True)
    
    # Save .npy format
    npy_file = os.path.join(npy_dir, f"displacement_field_{index}.npy")
    np.save(npy_file, displacement_field)
    
    # Calculate coordinate information
    h, w = displacement_field.shape[:2]
    
    # Create coordinate grids
    if roi_rect is not None:
        xmin, ymin, xmax, ymax = roi_rect
        # Create reference coordinates (same for all time points)
        y_coords, x_coords = np.mgrid[ymin:ymin+h, xmin:xmin+w]
        X_ref = x_coords.astype(np.float64)
        Y_ref = y_coords.astype(np.float64)
        
        # Calculate current coordinates by applying displacement
        u = displacement_field[:, :, 0]
        v = displacement_field[:, :, 1]
        X_current = X_ref + u
        Y_current = Y_ref + v
        
        # Apply ROI mask to set invalid regions to NaN
        if roi_mask is not None:
            # Crop ROI mask to match displacement field size
            roi_mask_crop = roi_mask[ymin:ymax, xmin:xmax]
            invalid_mask = ~roi_mask_crop
            
            X_ref[invalid_mask] = np.nan
            Y_ref[invalid_mask] = np.nan
            X_current[invalid_mask] = np.nan
            Y_current[invalid_mask] = np.nan
    else:
        # Fallback: create coordinate grids without ROI information
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        X_ref = x_coords.astype(np.float64)
        Y_ref = y_coords.astype(np.float64)
        
        u = displacement_field[:, :, 0]
        v = displacement_field[:, :, 1]
        X_current = X_ref + u
        Y_current = Y_ref + v
    
    # Save .mat format with coordinate information
    mat_file = os.path.join(mat_dir, f"displacement_field_{index}.mat")
    # Separate U and V components
    u = displacement_field[:, :, 0]
    v = displacement_field[:, :, 1]
    
    # Save all data including coordinates
    sio.savemat(mat_file, {
        'U': u, 
        'V': v,
        'X_ref': X_ref,
        'Y_ref': Y_ref,
        'X_current': X_current,
        'Y_current': Y_current
    })

def smooth_displacement_field(displacement_field, sigma=2.0):
    """
    Fast Gaussian smoothing of displacement field
    
    Args:
        displacement_field: Displacement field data with shape (H, W, 2)
        sigma: Gaussian filter standard deviation, controls smoothing level
    """
    from scipy.ndimage import gaussian_filter
    import numpy as np
    
    # Directly process U and V directions
    smoothed = np.zeros_like(displacement_field)
    
    for i in range(2):
        component = displacement_field[..., i]
        valid_mask = ~np.isnan(component)
        
        if not np.any(valid_mask):
            smoothed[..., i] = component
            continue
            
        # Fill invalid areas with 0
        filled = np.where(valid_mask, component, 0)
        
        # Complete Gaussian filtering in one step
        smoothed_data = gaussian_filter(filled, sigma)
        weight = gaussian_filter(valid_mask.astype(float), sigma)
        
        # Normalize and restore NaN
        with np.errstate(divide='ignore', invalid='ignore'):
            smoothed[..., i] = np.where(weight > 0.01, 
                                      smoothed_data / weight, 
                                      np.nan)
    
    return smoothed

def process_image_pair(ref_img, def_img, project_root, model, device, 
                      maxDisplacement=50, roi_mask=None,
                      use_smooth=True, sigma=2.0):
    """
    Process entire image pair without cropping
    
    Args:
        ref_img: Reference image
        def_img: Deformed image
        project_root: Project root directory
        model: RAFT model
        device: Computation device
        maxDisplacement: Maximum displacement
        roi_mask: ROI mask, binary array same size as input image
        use_smooth: Whether to use smoothing
        sigma: Gaussian smoothing sigma parameter
    """
    start_total = time.time()  # Start timing total process
    H, W, _ = ref_img.shape
    
    # If ROI mask provided, ensure it's boolean type
    if roi_mask is not None:
        roi_mask = roi_mask.astype(bool)
    else:
        # If no mask provided, create all True mask
        roi_mask = np.ones((H, W), dtype=bool)

    # Add RAFT inference time statistics
    start_inference = time.time()
    flow_low, flow_up = inference(model, ref_img, def_img, device, test_mode=True)
    flow_up = flow_up.squeeze(0)
    inference_time = time.time() - start_inference
    
    # Convert flow to numpy array
    u = flow_up[0].cpu().numpy()
    v = flow_up[1].cpu().numpy()
    displacement_field = np.stack((u, v), axis=-1)
    
    # Apply ROI mask
    displacement_field[~roi_mask] = np.nan
    
    # Apply smoothing if requested
    if use_smooth:
        displacement_field = smooth_displacement_field(displacement_field, sigma=sigma)
    
    total_time = time.time() - start_total
    
    print(f"\nTime statistics:")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"RAFT inference time: {inference_time:.2f} seconds")
    print(f"RAFT inference percentage: {(inference_time/total_time*100):.1f}%")
    print(f"Other operations time: {(total_time-inference_time):.2f} seconds")
    
    return displacement_field, None  # Return None for windows as they're not used