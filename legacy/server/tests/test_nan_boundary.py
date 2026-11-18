"""Regression test: boundary contamination in warp_displacement_field."""

import numpy as np
from raft_dic_gui.incremental import warp_displacement_field


class TestBoundaryContamination:

    def test_no_contamination_at_valid_boundary(self):
        """Boundary pixels of delta_u should not be pulled toward zero."""
        H, W = 100, 100

        # delta_u: valid block [30:70, 30:70] with uniform value 3.0
        delta_u = np.full((H, W, 2), np.nan)
        delta_u[30:70, 30:70, :] = 3.0

        # accumulated_u: uniform 9.7px rightward shift
        accumulated_u = np.full((H, W, 2), np.nan)
        accumulated_u[20:60, 20:60, 0] = 9.7   # u (x-shift)
        accumulated_u[20:60, 20:60, 1] = 0.0    # v (no y-shift)

        result = warp_displacement_field(delta_u, accumulated_u)

        # All valid sampled values should be close to 3.0, NOT pulled toward 0
        valid = ~np.isnan(result[..., 0])
        valid_values = result[valid, 0]

        assert len(valid_values) > 0, "Should have some valid samples"
        np.testing.assert_allclose(
            valid_values, 3.0, atol=0.05,
            err_msg="Boundary contamination detected: values deviate from 3.0"
        )
