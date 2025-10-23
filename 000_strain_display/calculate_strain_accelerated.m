function [exx, eyy, exy, theta_deg] = calculate_strain_accelerated(X, Y, U, V, window_size, use_parallel, step)
% calculate_strain_accelerated
% Compute strain components exx, eyy, exy on a downsampled grid using
% local linear fitting within a window, then interpolate to full grids.

    if nargin < 7 || isempty(step), step = 1; end
    if nargin < 6 || isempty(use_parallel), use_parallel = false; end
    if nargin < 5 || isempty(window_size), window_size = 21; end

    [m, n] = size(X);
    hw = floor(window_size / 2);

    % Downsampled indices (avoid borders by half window)
    row_indices = (1+hw):step:(m-hw);
    col_indices = (1+hw):step:(n-hw);
    [J_grid, I_grid] = meshgrid(col_indices, row_indices);
    calc_points_indices = sub2ind([m,n], I_grid(:), J_grid(:));

    num_calc_points = length(calc_points_indices);
    exx_sparse = NaN(num_calc_points, 1);
    eyy_sparse = NaN(num_calc_points, 1);
    exy_sparse = NaN(num_calc_points, 1);
    theta_sparse = NaN(num_calc_points, 1); % degrees

    if use_parallel
        parfor k = 1:num_calc_points %#ok<PFBNS>
            [exx_sparse(k), eyy_sparse(k), exy_sparse(k)] = ...
                calc_strain_at_point(k, calc_points_indices, X, Y, U, V, hw);
            % Rotation from polar decomposition of F = I + GradU
            if ~isnan(exx_sparse(k))
                [i, j] = ind2sub([m,n], calc_points_indices(k));
                % Local linear fits were around (i,j); reuse gradients via direct finite neighborhood could be noisy,
                % so approximate GradU using local plane coefficients already obtained implicitly.
                % To stay consistent, recompute quick LSQ at this center point only.
                try
                    ii = i-hw:i+hw; jj = j-hw:j+hw;
                    x_local = X(ii, jj); y_local = Y(ii, jj);
                    u_local = U(ii, jj); v_local = V(ii, jj);
                    x_vec = x_local(:); y_vec = y_local(:);
                    u_vec = u_local(:); v_vec = v_local(:);
                    valid_pts = isfinite(x_vec) & isfinite(y_vec) & isfinite(u_vec) & isfinite(v_vec);
                    if nnz(valid_pts) >= 3
                        A = [x_vec(valid_pts), y_vec(valid_pts), ones(nnz(valid_pts),1)];
                        p_u = A \ u_vec(valid_pts);
                        p_v = A \ v_vec(valid_pts);
                        dU_dx = p_u(1); dU_dy = p_u(2);
                        dV_dx = p_v(1); dV_dy = p_v(2);
                        F = eye(2) + [dU_dx, dU_dy; dV_dx, dV_dy];
                        [Uq,~,Vq] = svd(F);
                        R = Uq*Vq';
                        theta_sparse(k) = rad2deg(atan2(R(2,1), R(1,1)));
                    end
                catch
                end
            end
        end
    else
        for k = 1:num_calc_points
            [exx_sparse(k), eyy_sparse(k), exy_sparse(k)] = ...
                calc_strain_at_point(k, calc_points_indices, X, Y, U, V, hw);
            if ~isnan(exx_sparse(k))
                [i, j] = ind2sub([m,n], calc_points_indices(k));
                try
                    ii = i-hw:i+hw; jj = j-hw:j+hw;
                    x_local = X(ii, jj); y_local = Y(ii, jj);
                    u_local = U(ii, jj); v_local = V(ii, jj);
                    x_vec = x_local(:); y_vec = y_local(:);
                    u_vec = u_local(:); v_vec = v_local(:);
                    valid_pts = isfinite(x_vec) & isfinite(y_vec) & isfinite(u_vec) & isfinite(v_vec);
                    if nnz(valid_pts) >= 3
                        A = [x_vec(valid_pts), y_vec(valid_pts), ones(nnz(valid_pts),1)];
                        p_u = A \ u_vec(valid_pts);
                        p_v = A \ v_vec(valid_pts);
                        dU_dx = p_u(1); dU_dy = p_u(2);
                        dV_dx = p_v(1); dV_dy = p_v(2);
                        F = eye(2) + [dU_dx, dU_dy; dV_dx, dV_dy];
                        [Uq,~,Vq] = svd(F);
                        R = Uq*Vq';
                        theta_sparse(k) = rad2deg(atan2(R(2,1), R(1,1)));
                    end
                catch
                end
            end
        end
    end

    % Keep valid results
    valid_calc_indices = ~isnan(exx_sparse);
    x_calc = X(calc_points_indices(valid_calc_indices));
    y_calc = Y(calc_points_indices(valid_calc_indices));

    if isempty(x_calc)
        exx = NaN(m,n); eyy = NaN(m,n); exy = NaN(m,n); theta_deg = NaN(m,n);
        return;
    end

    exx_valid = exx_sparse(valid_calc_indices);
    eyy_valid = eyy_sparse(valid_calc_indices);
    exy_valid = exy_sparse(valid_calc_indices);
    theta_valid = theta_sparse(valid_calc_indices);

    % Interpolate sparse results back to full grid
    F_exx = scatteredInterpolant(x_calc, y_calc, exx_valid, 'linear', 'none');
    F_eyy = scatteredInterpolant(x_calc, y_calc, eyy_valid, 'linear', 'none');
    F_exy = scatteredInterpolant(x_calc, y_calc, exy_valid, 'linear', 'none');
    F_theta = scatteredInterpolant(x_calc, y_calc, theta_valid, 'linear', 'none');

    exx = F_exx(X, Y);
    eyy = F_eyy(X, Y);
    exy = F_exy(X, Y);
    theta_deg = F_theta(X, Y);
end
