# Phase 4: Scientific Analysis & Instrument Refinement

## Overview
This phase focuses on transforming RAFT-DIC from a displacement tracker into a precision scientific instrument. The core achievement is the implementation of a robust **Virtual Strain Gauge (VSG)** algorithm for strain calculation, replacing simple finite differences with advanced local polynomial fitting.

## Key Implementations

### 1. Advanced Strain Calculation (VSG)
We replaced the standard central difference method with a **Local Polynomial Least Squares** approach, often called the Virtual Strain Gauge (VSG) method.

*   **Methodology**:
    *   For each pixel, a local window (size $N \times N$) is selected.
    *   A polynomial surface (1st or 2nd order) is fitted to the displacement data $(u, v)$ within this window using weighted least squares.
    *   The coefficients of the fitted polynomial directly yield the displacement gradients ($\frac{\partial u}{\partial x}, \frac{\partial u}{\partial y}$, etc.), which are used to calculate strain tensors.
*   **Advantages**:
    *   **Noise Suppression**: Uses all pixels in the window to "average out" noise, significantly superior to 2-point or 3-point differences.
    *   **Tunable Precision**: Users can trade off spatial resolution for strain resolution by adjusting the VSG Size.

### 2. Robust Boundary Handling (Strategy 2)
A critical improvement was addressing edge artifacts. Standard convolution methods fill invalid regions (outside ROI) with zeros, causing severe errors at boundaries. We implemented a **Vectorized Weighted Least Squares** strategy:

*   **Mechanism**:
    *   The algorithm dynamically constructs the normal equations ($M\mathbf{a} = \mathbf{b}$) for every pixel.
    *   **Validity Masking**: Invalid pixels (NaNs or masked regions) are assigned a weight of 0.
    *   **Dynamic Fit**: The least squares fit is performed using *only* the valid pixels remaining in the window.
*   **Result**:
    *   Strain can be calculated right up to the edge of the ROI without "roll-off" artifacts.
    *   No data loss (erosion) at boundaries.

### 3. Performance Optimization
Despite the increased complexity of per-pixel least squares, we maintained high performance:
*   **Vectorized Construction**: The $M$ and $\mathbf{b}$ matrices are constructed for the entire image at once using `scipy.ndimage.correlate`.
*   **Batch Solving**: The linear systems are solved in parallel using `np.linalg.solve` (with explicit reshaping to handle broadcasting correctly).
*   **Step Parameter**: Added a `Step` (stride) parameter to allow downsampling the calculation grid, reducing memory usage and computation time for large datasets.

### 4. User Interface Updates
The Post-Processing panel was updated to expose these new capabilities:
*   **VSG Size**: Odd integer (9-101 px).
*   **Polynomial Order**: 1 (Linear) or 2 (Quadratic).
*   **Weighting**: Uniform or Gaussian (Gaussian gives higher weight to central pixels).
*   **Step**: Calculation stride for efficiency.

## Next Steps
*   **Virtual Extensometers**: Implement Point, Line, and Area probes for time-series analysis.
*   **Data Export**: Enable exporting the calculated strain fields and probe data.
