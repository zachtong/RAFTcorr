"""Deep tests for strain route endpoints: validation, status, render, download."""

import numpy as np
import pytest

from server.session import session


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_strain_results(num_frames=3, h=64, w=64):
    """Build mock strain results with all standard components."""
    components = [
        "exx", "eyy", "exy", "e1", "e2",
        "max_shear", "von_mises", "rotation",
        "dexx_dt", "deyy_dt", "dexy_dt",
    ]
    results = []
    for _ in range(num_frames):
        strain_dict = {}
        for comp in components:
            arr = np.random.randn(h, w).astype(np.float32) * 0.01
            if comp in ("max_shear", "von_mises"):
                arr = np.abs(arr)
            strain_dict[comp] = arr
        results.append(strain_dict)
    return results, components


@pytest.fixture()
def mock_strain_session(mock_displacement_results):
    """Populate session with displacement AND strain results."""
    results, components = _make_strain_results()
    session.strain_results = results
    session.strain_components = components
    return results


# ===========================================================================
# 1. Input Validation Tests for POST /api/strain/calculate
# ===========================================================================

class TestCalculateValidation:
    """Validate that /calculate rejects bad parameters with 400."""

    def test_no_displacement_returns_400(self, client):
        """Calculate without displacement results should fail."""
        resp = client.post("/api/strain/calculate", json={})
        assert resp.status_code == 400
        assert "error" in resp.get_json()

    def test_invalid_method(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"method": "invalid_method"})
        assert resp.status_code == 400
        data = resp.get_json()
        assert "method" in data["error"].lower()

    def test_valid_method_green_lagrange(self, client, mock_displacement_results):
        """'green_lagrange' should pass validation and return 200."""
        resp = client.post("/api/strain/calculate", json={"method": "green_lagrange"})
        assert resp.status_code == 200

    def test_valid_method_engineering(self, client, mock_displacement_results):
        """'engineering' should pass validation and return 200."""
        resp = client.post("/api/strain/calculate", json={"method": "engineering"})
        assert resp.status_code == 200

    def test_vsg_even_rejected(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"vsg_size": 10})
        assert resp.status_code == 400
        assert "odd" in resp.get_json()["error"].lower()

    def test_vsg_too_small(self, client, mock_displacement_results):
        """vsg_size must be >= 9."""
        resp = client.post("/api/strain/calculate", json={"vsg_size": 7})
        assert resp.status_code == 400
        assert "9" in resp.get_json()["error"]

    def test_vsg_non_integer(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"vsg_size": "abc"})
        assert resp.status_code == 400

    def test_vsg_negative(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"vsg_size": -5})
        assert resp.status_code == 400

    def test_step_zero_rejected(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"step": 0})
        assert resp.status_code == 400

    def test_step_negative_rejected(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"step": -1})
        assert resp.status_code == 400

    def test_poly_order_zero_rejected(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"poly_order": 0})
        assert resp.status_code == 400

    def test_invalid_weighting(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"weighting": "hamming"})
        assert resp.status_code == 400
        assert "weighting" in resp.get_json()["error"].lower()

    def test_negative_temporal_sigma(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"temporal_sigma": -1.0})
        assert resp.status_code == 400

    def test_negative_spatial_sigma(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"spatial_sigma": -0.5})
        assert resp.status_code == 400

    def test_fps_zero_rejected(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"fps": 0})
        assert resp.status_code == 400

    def test_fps_negative_rejected(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"fps": -10})
        assert resp.status_code == 400

    def test_cond_threshold_zero_rejected(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"cond_threshold": 0})
        assert resp.status_code == 400

    def test_cond_threshold_negative_rejected(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"cond_threshold": -100})
        assert resp.status_code == 400


# ===========================================================================
# 2. Parameter Combination Tests
# ===========================================================================

class TestCalculateParameterCombinations:
    """Test that various valid parameter combinations are accepted (200)."""

    def test_green_lagrange_gaussian_custom_sigma(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={
            "method": "green_lagrange",
            "weighting": "Gaussian",
            "spatial_sigma": 2.0,
        })
        assert resp.status_code == 200

    def test_engineering_uniform(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={
            "method": "engineering",
            "weighting": "uniform",
        })
        assert resp.status_code == 200

    def test_vsg_9(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"vsg_size": 9})
        assert resp.status_code == 200

    def test_vsg_15(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"vsg_size": 15})
        assert resp.status_code == 200

    def test_vsg_31(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"vsg_size": 31})
        assert resp.status_code == 200

    def test_poly_order_1(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"poly_order": 1})
        assert resp.status_code == 200

    def test_poly_order_2(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"poly_order": 2})
        assert resp.status_code == 200

    def test_boundary_erosion_auto(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"boundary_erosion": -1})
        assert resp.status_code == 200

    def test_boundary_erosion_explicit(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"boundary_erosion": 5})
        assert resp.status_code == 200

    def test_temporal_sigma_positive(self, client, mock_displacement_results):
        resp = client.post("/api/strain/calculate", json={"temporal_sigma": 1.5})
        assert resp.status_code == 200

    def test_all_defaults(self, client, mock_displacement_results):
        """Empty JSON body should use all defaults and succeed."""
        resp = client.post("/api/strain/calculate", json={})
        assert resp.status_code == 200


# ===========================================================================
# 3. Conflict: calculate while already computing
# ===========================================================================

class TestCalculateConflict:
    """Calculate endpoint should return 409 if already computing."""

    def test_duplicate_calculate_returns_409(self, client, mock_displacement_results):
        session.strain_computing = True
        resp = client.post("/api/strain/calculate", json={})
        assert resp.status_code == 409
        assert "already in progress" in resp.get_json()["error"].lower()


# ===========================================================================
# 4. Status Endpoint Tests
# ===========================================================================

class TestStatus:
    """GET /api/strain/status should report correct state."""

    def test_status_idle_no_results(self, client):
        # Ensure no results leaked from async threads in prior tests
        session.strain_results = []
        session.strain_computing = False
        session.strain_components = []
        resp = client.get("/api/strain/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["computed"] is False
        assert data["computing"] is False
        assert data["components"] == []
        assert data["num_frames"] == 0

    def test_status_with_results(self, client, mock_strain_session):
        resp = client.get("/api/strain/status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["computed"] is True
        assert data["computing"] is False
        assert data["num_frames"] == 3
        assert "exx" in data["components"]
        assert "von_mises" in data["components"]

    def test_status_fields_present(self, client):
        resp = client.get("/api/strain/status")
        data = resp.get_json()
        for key in ("computed", "computing", "components", "num_frames"):
            assert key in data, f"Missing key: {key}"

    def test_status_computing_flag(self, client):
        """When strain_computing is True, status reflects it."""
        session.strain_computing = True
        resp = client.get("/api/strain/status")
        assert resp.get_json()["computing"] is True


# ===========================================================================
# 5. Frame Data Endpoint (GET /api/strain/frame/<idx>)
# ===========================================================================

class TestFrameData:
    """GET /api/strain/frame/<idx> — raw strain data as JSON."""

    def test_no_results_returns_404(self, client):
        # Ensure no strain results leaked from async threads in prior tests
        session.strain_results = []
        resp = client.get("/api/strain/frame/0")
        assert resp.status_code == 404

    def test_negative_index_not_routable(self, client, mock_strain_session):
        """Flask <int:idx> does not match negative values -> 404."""
        resp = client.get("/api/strain/frame/-1")
        assert resp.status_code == 404

    def test_out_of_range_index(self, client, mock_strain_session):
        resp = client.get("/api/strain/frame/99")
        assert resp.status_code == 400

    def test_valid_frame_default_component(self, client, mock_strain_session):
        resp = client.get("/api/strain/frame/0")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "shape" in data
        assert data["shape"] == [64, 64]

    def test_all_components(self, client, mock_strain_session):
        """Every component in the strain dict should be retrievable."""
        for comp in session.strain_components:
            resp = client.get(f"/api/strain/frame/0?component={comp}")
            assert resp.status_code == 200, f"Component '{comp}' failed"
            assert resp.get_json()["shape"] == [64, 64]

    def test_invalid_component(self, client, mock_strain_session):
        resp = client.get("/api/strain/frame/0?component=nonexistent")
        assert resp.status_code == 400
        assert "nonexistent" in resp.get_json()["error"]

    def test_custom_vmin_vmax(self, client, mock_strain_session):
        resp = client.get("/api/strain/frame/0?component=exx&vmin=-0.01&vmax=0.01")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["vmin"] == -0.01
        assert data["vmax"] == 0.01

    def test_last_frame_accessible(self, client, mock_strain_session):
        resp = client.get("/api/strain/frame/2")
        assert resp.status_code == 200


# ===========================================================================
# 6. Range Endpoint (GET /api/strain/range/<idx>)
# ===========================================================================

class TestRange:
    """GET /api/strain/range/<idx> — auto vmin/vmax for a component."""

    def test_no_results_returns_404(self, client):
        session.strain_results = []
        resp = client.get("/api/strain/range/0")
        assert resp.status_code == 404

    def test_out_of_range_index(self, client, mock_strain_session):
        resp = client.get("/api/strain/range/99")
        assert resp.status_code == 400

    def test_valid_range(self, client, mock_strain_session):
        resp = client.get("/api/strain/range/0?component=exx")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "vmin" in data
        assert "vmax" in data
        assert isinstance(data["vmin"], float)
        assert isinstance(data["vmax"], float)
        assert data["vmin"] <= data["vmax"]

    def test_invalid_component(self, client, mock_strain_session):
        resp = client.get("/api/strain/range/0?component=bogus")
        assert resp.status_code == 400

    def test_all_components_have_range(self, client, mock_strain_session):
        for comp in session.strain_components:
            resp = client.get(f"/api/strain/range/0?component={comp}")
            assert resp.status_code == 200, f"Range failed for '{comp}'"
            data = resp.get_json()
            assert data["vmin"] <= data["vmax"]


# ===========================================================================
# 7. Render Endpoint (GET /api/strain/render/<idx>)
# ===========================================================================

class TestRender:
    """GET /api/strain/render/<idx> — rendered strain overlay as PNG."""

    def test_no_results_returns_404(self, client):
        session.strain_results = []
        resp = client.get("/api/strain/render/0")
        assert resp.status_code == 404

    def test_negative_index_not_routable(self, client, mock_strain_session):
        """Flask <int:idx> does not match negative values -> 404."""
        resp = client.get("/api/strain/render/-1")
        assert resp.status_code == 404

    def test_out_of_range_index(self, client, mock_strain_session):
        resp = client.get("/api/strain/render/99")
        assert resp.status_code == 400

    def test_default_render(self, client, mock_strain_session):
        resp = client.get("/api/strain/render/0")
        assert resp.status_code == 200
        assert "image/" in resp.content_type

    def test_render_all_components(self, client, mock_strain_session):
        for comp in ("exx", "eyy", "exy", "von_mises", "rotation"):
            resp = client.get(f"/api/strain/render/0?component={comp}")
            assert resp.status_code == 200, f"Render failed for '{comp}'"
            assert "image/" in resp.content_type

    def test_render_invalid_component(self, client, mock_strain_session):
        resp = client.get("/api/strain/render/0?component=fake_comp")
        assert resp.status_code == 400

    def test_render_custom_colormap(self, client, mock_strain_session):
        resp = client.get("/api/strain/render/0?component=exx&colormap=jet")
        assert resp.status_code == 200
        assert "image/" in resp.content_type

    def test_render_custom_vmin_vmax(self, client, mock_strain_session):
        resp = client.get("/api/strain/render/0?component=exx&vmin=-0.01&vmax=0.01")
        assert resp.status_code == 200

    def test_render_overlay_only(self, client, mock_strain_session):
        resp = client.get("/api/strain/render/0?component=exx&overlay_only=true")
        assert resp.status_code == 200
        assert "image/" in resp.content_type

    def test_render_overlay_data_mode(self, client, mock_strain_session):
        resp = client.get(
            "/api/strain/render/0?component=exx&overlay_only=true&render_mode=data"
        )
        assert resp.status_code == 200
        # Data texture mode includes X-Data-Min / X-Data-Max headers
        assert "X-Data-Min" in resp.headers
        assert "X-Data-Max" in resp.headers

    def test_render_with_viewport(self, client, mock_strain_session):
        resp = client.get("/api/strain/render/0?component=exx&vw=128&vh=128")
        assert resp.status_code == 200

    def test_render_log_scale(self, client, mock_strain_session):
        resp = client.get("/api/strain/render/0?component=exx&log_scale=true")
        assert resp.status_code == 200

    def test_render_returns_png_bytes(self, client, mock_strain_session):
        resp = client.get("/api/strain/render/0?component=exx")
        # PNG magic number
        assert resp.data[:4] == b"\x89PNG"

    def test_render_caches_second_call(self, client, mock_strain_session):
        """Two identical requests should return the same content (cached)."""
        url = "/api/strain/render/0?component=exx&colormap=turbo"
        r1 = client.get(url)
        r2 = client.get(url)
        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r1.data == r2.data

    def test_render_last_frame(self, client, mock_strain_session):
        resp = client.get("/api/strain/render/2?component=exx")
        assert resp.status_code == 200


# ===========================================================================
# 8. Download Endpoint (GET /api/strain/download/<idx>)
# ===========================================================================

class TestDownload:
    """GET /api/strain/download/<idx> — high-res PNG download via matplotlib."""

    def test_no_results_returns_404(self, client):
        session.strain_results = []
        resp = client.get("/api/strain/download/0")
        assert resp.status_code == 404

    def test_negative_index_not_routable(self, client, mock_strain_session):
        """Flask <int:idx> does not match negative values -> 404."""
        resp = client.get("/api/strain/download/-1")
        assert resp.status_code == 404

    def test_out_of_range_index(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/99")
        assert resp.status_code == 400

    def test_invalid_component(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/0?component=fake")
        assert resp.status_code == 400

    def test_default_download(self, client, mock_strain_session):
        """Default download should return a PNG attachment."""
        resp = client.get("/api/strain/download/0")
        assert resp.status_code == 200
        assert "image/png" in resp.content_type
        disp_header = resp.headers.get("Content-Disposition", "")
        assert "attachment" in disp_header
        assert ".png" in disp_header

    def test_download_filename_contains_component(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/0?component=eyy")
        assert resp.status_code == 200
        disp_header = resp.headers.get("Content-Disposition", "")
        assert "eyy" in disp_header

    def test_download_custom_dpi(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/0?dpi=100")
        assert resp.status_code == 200

    def test_download_dpi_out_of_range_low(self, client, mock_strain_session):
        """DPI below 50 should be rejected."""
        resp = client.get("/api/strain/download/0?dpi=10")
        assert resp.status_code == 400

    def test_download_dpi_out_of_range_high(self, client, mock_strain_session):
        """DPI above 600 should be rejected."""
        resp = client.get("/api/strain/download/0?dpi=1000")
        assert resp.status_code == 400

    def test_download_boundary_dpi_50(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/0?dpi=50")
        assert resp.status_code == 200

    def test_download_boundary_dpi_600(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/0?dpi=600")
        assert resp.status_code == 200

    def test_download_custom_colormap(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/0?component=exx&colormap=viridis")
        assert resp.status_code == 200

    def test_download_custom_alpha(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/0?alpha=0.5")
        assert resp.status_code == 200

    def test_download_log_scale(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/0?component=von_mises&log_scale=true&vmin=0.001&vmax=0.1")
        assert resp.status_code == 200

    def test_download_custom_vmin_vmax(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/0?component=exx&vmin=-0.005&vmax=0.005")
        assert resp.status_code == 200

    def test_download_multiple_components(self, client, mock_strain_session):
        """Each component should produce a downloadable image."""
        for comp in ("exx", "eyy", "exy", "von_mises", "rotation"):
            resp = client.get(f"/api/strain/download/0?component={comp}")
            assert resp.status_code == 200, f"Download failed for '{comp}'"

    def test_download_last_frame(self, client, mock_strain_session):
        resp = client.get("/api/strain/download/2?component=exx")
        assert resp.status_code == 200


# ===========================================================================
# 9. Edge Cases
# ===========================================================================

class TestEdgeCases:
    """Edge cases and boundary conditions for strain endpoints."""

    def test_frame_data_with_none_strain_dict(self, client, mock_displacement_results):
        """If a strain_dict entry is None, endpoints should handle gracefully."""
        session.strain_results = [None, None, None]
        session.strain_components = ["exx"]
        resp = client.get("/api/strain/frame/0?component=exx")
        assert resp.status_code == 400

    def test_render_with_none_strain_dict(self, client, mock_displacement_results):
        session.strain_results = [None, None, None]
        session.strain_components = ["exx"]
        resp = client.get("/api/strain/render/0?component=exx")
        assert resp.status_code == 400

    def test_range_with_none_strain_dict(self, client, mock_displacement_results):
        session.strain_results = [None, None, None]
        session.strain_components = ["exx"]
        resp = client.get("/api/strain/range/0?component=exx")
        assert resp.status_code == 400

    def test_frame_boundary_idx_zero(self, client, mock_strain_session):
        resp = client.get("/api/strain/frame/0")
        assert resp.status_code == 200

    def test_frame_boundary_idx_last(self, client, mock_strain_session):
        resp = client.get("/api/strain/frame/2")
        assert resp.status_code == 200

    def test_frame_boundary_idx_exactly_at_length(self, client, mock_strain_session):
        """Index equal to num_frames should be out of range."""
        resp = client.get("/api/strain/frame/3")
        assert resp.status_code == 400

    def test_render_with_all_nan_data(self, client, mock_displacement_results):
        """Strain data that is all-NaN should still render without crashing."""
        nan_arr = np.full((64, 64), np.nan, dtype=np.float32)
        session.strain_results = [{"exx": nan_arr}]
        session.strain_components = ["exx"]
        resp = client.get("/api/strain/render/0?component=exx")
        assert resp.status_code == 200

    def test_download_with_no_reference_image(self, client):
        """Download with missing reference image should return 500."""
        results, components = _make_strain_results(num_frames=1)
        session.strain_results = results
        session.strain_components = components
        session.image_width = 64
        session.image_height = 64
        session.reference_image = None
        resp = client.get("/api/strain/download/0")
        assert resp.status_code == 500
