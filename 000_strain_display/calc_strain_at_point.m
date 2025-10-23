function [exx_p, eyy_p, exy_p] = calc_strain_at_point(k, indices, X, Y, U, V, hw)
% calc_strain_at_point
% Fit local linear models for U(x,y) and V(x,y) in a (2*hw+1)^2 window
% around the k-th downsampled point and return exx, eyy, exy at that point.

    exx_p = NaN; eyy_p = NaN; exy_p = NaN;

    [i, j] = ind2sub(size(X), indices(k));
    if isnan(X(i,j))
        return;
    end

    x_local = X(i-hw:i+hw, j-hw:j+hw);
    y_local = Y(i-hw:i+hw, j-hw:j+hw);
    u_local = U(i-hw:i+hw, j-hw:j+hw);
    v_local = V(i-hw:i+hw, j-hw:j+hw);

    x_vec = x_local(:); y_vec = y_local(:);
    u_vec = u_local(:); v_vec = v_local(:);

    valid_pts = ~isnan(x_vec) & ~isnan(u_vec) & ~isnan(v_vec);
    if sum(valid_pts) < 3
        return;
    end

    x_fit = x_vec(valid_pts); y_fit = y_vec(valid_pts);
    u_fit = u_vec(valid_pts); v_fit = v_vec(valid_pts);

    A = [x_fit, y_fit, ones(size(x_fit))];

    p_u = A \ u_fit;
    p_v = A \ v_fit;

    dU_dx = p_u(1); dU_dy = p_u(2);
    dV_dx = p_v(1); dV_dy = p_v(2);

    exx_p = dU_dx;
    eyy_p = dV_dy;
    exy_p = 0.5 * (dU_dy + dV_dx);
end

