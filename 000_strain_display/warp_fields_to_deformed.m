function [U_d, V_d, exx_d, eyy_d, exy_d, Xq, Yq, theta_d, ROI_d] = warp_fields_to_deformed(X, Y, U, V, exx, eyy, exy, img_h, img_w, theta)
% warp_fields_to_deformed
% Forward-map fields defined on the reference grid (X,Y) to the deformed
% image pixel grid using scatteredInterpolant with shared triangulation.
% Returns warped fields and the query grid Xq,Yq (pixel centers).

    [m, n] = size(X);
    Xd = X + U; Yd = Y + V;

    % Query grid is the native pixel grid of the deformed image
    if nargin < 8 || isempty(img_h) || isempty(img_w)
        img_h = m; img_w = n; % fallback to ROI size (may appear smaller on full image)
    end
    [Xq, Yq] = meshgrid(1:img_w, 1:img_h);
    theta_d = [];
    ROI_d = false(img_h, img_w);

    % Vectorized data
    Xd_v = Xd(:); Yd_v = Yd(:);
    U_v  = U(:);  V_v  = V(:);
    exx_v = exx(:); eyy_v = eyy(:); exy_v = exy(:);

    % Build conservative deformed-domain ROI by forward stamping to preserve holes
    ROI_ref = isfinite(X) & isfinite(Y) & isfinite(U) & isfinite(V);
    if any(ROI_ref(:))
        xd = X(ROI_ref) + U(ROI_ref);
        yd = Y(ROI_ref) + V(ROI_ref);
        xpi = round(xd); ypi = round(yd);
        inb = xpi >= 1 & xpi <= img_w & ypi >= 1 & ypi <= img_h & isfinite(xpi) & isfinite(ypi);
        if any(inb)
            ind = sub2ind([img_h, img_w], ypi(inb), xpi(inb));
            ROI_d(:) = false; ROI_d(ind) = true;
        end
    end

    % Interpolate displacements (use only coords + U,V finite)
    valid_uv = isfinite(Xd_v) & isfinite(Yd_v) & isfinite(U_v) & isfinite(V_v);
    if nnz(valid_uv) >= 3
        Xp = Xd_v(valid_uv); Yp = Yd_v(valid_uv);
        F_uv = scatteredInterpolant(Xp, Yp, U_v(valid_uv), 'linear', 'none');
        U_d = F_uv(Xq, Yq);
        F_uv.Values = V_v(valid_uv);
        V_d = F_uv(Xq, Yq);
    else
        U_d = NaN(m,n); V_d = NaN(m,n);
    end

    % Interpolate strains (use only coords + exx,eyy,exy finite)
    valid_e = isfinite(Xd_v) & isfinite(Yd_v) & isfinite(exx_v) & isfinite(eyy_v) & isfinite(exy_v);
    if nnz(valid_e) >= 3
        Xp = Xd_v(valid_e); Yp = Yd_v(valid_e);
        F_e = scatteredInterpolant(Xp, Yp, exx_v(valid_e), 'linear', 'none');
        exx_d = F_e(Xq, Yq);
        F_e.Values = eyy_v(valid_e);
        eyy_d = F_e(Xq, Yq);
        F_e.Values = exy_v(valid_e);
        exy_d = F_e(Xq, Yq);
    else
        exx_d = NaN(m,n); eyy_d = NaN(m,n); exy_d = NaN(m,n);
    end

    % Optional: interpolate theta if provided (degrees)
    if nargin >= 10 && ~isempty(theta)
        theta_v = theta(:);
        valid_t = isfinite(Xd_v) & isfinite(Yd_v) & isfinite(theta_v);
        if nnz(valid_t) >= 3
            Xp = Xd_v(valid_t); Yp = Yd_v(valid_t);
            F_t = scatteredInterpolant(Xp, Yp, theta_v(valid_t), 'linear', 'none');
            theta_d = F_t(Xq, Yq);
        else
            theta_d = NaN(m,n);
        end
    end
end
