function [figs, names] = plot_strains(X, Y, exx, eyy, exy, e_vm, ref_rgb, config, frame_index, fields, varargin)
% plot_strains
% Render strain fields as heatmaps aligned to the provided grid using
% imagesc with ROI-based AlphaData. Supports optional theta (rotation).

    if nargin < 9, frame_index = []; end
    if nargin < 10 || isempty(fields), fields = {'exx','eyy','exy','e_vm'}; end
    theta = [];
    if ~isempty(varargin)
        theta = varargin{1};
    end
    cm = config.colormap_name; if isempty(cm), cm = 'parula'; end

    figs = gobjects(0);
    names = {};
    vis = 'on'; if isfield(config,'show_figures') && ~config.show_figures, vis = 'off'; end

    % Bounds from X,Y
    finiteXY = isfinite(X) & isfinite(Y);
    if any(finiteXY(:))
        xmin = min(X(finiteXY)); xmax = max(X(finiteXY));
        ymin = min(Y(finiteXY)); ymax = max(Y(finiteXY));
    else
        [m,n] = size(exx);
        xmin = 0.5; xmax = n+0.5; ymin = 0.5; ymax = m+0.5;
    end
    xdata = [xmin-0.5, xmax+0.5];
    ydata = [ymin-0.5, ymax+0.5];

    % Helper to render one field
    function renderOne(Z, fieldName, titleText, cbarLabel)
        figs(end+1) = figure('Name', titleWithFrame(titleText, frame_index), 'Visible', vis); %#ok<AGROW>
        if ~isempty(ref_rgb), imshow(ref_rgb, 'InitialMagnification','fit'); hold on; end
        ROI = isfinite(Z) & finiteXY;
        if isfield(config,'alpha_mask') && ~isempty(config.alpha_mask) && all(size(config.alpha_mask) == size(Z))
            ROI = ROI & logical(config.alpha_mask);
        end
        h = imagesc(xdata, ydata, Z);
        set(h, 'AlphaData', double(ROI) * config.overlay_alpha);
        try colormap(cm); catch, colormap('parula'); end
        if isFixed(config, fieldName) && isfield(config.color_limits, fieldName)
            caxis(config.color_limits.(fieldName));
        end
        cb = colorbar; ylabel(cb, cbarLabel);
        axis image; set(gca,'YDir','reverse'); title(titleWithFrame(titleText, frame_index)); hold off;
        names{end+1} = fieldName; %#ok<AGROW>
    end

    if any(strcmpi(fields,'exx'))
        renderOne(exx, 'exx', 'exx Strain', 'e_{xx}');
    end
    if any(strcmpi(fields,'eyy'))
        renderOne(eyy, 'eyy', 'eyy Strain', 'e_{yy}');
    end
    if any(strcmpi(fields,'exy'))
        renderOne(exy, 'exy', 'exy Strain', 'e_{xy}');
    end
    if any(strcmpi(fields,'e_vm'))
        renderOne(e_vm, 'e_vm', 'Von Mises Strain', 'e_{vM}');
    end
    if any(strcmpi(fields,'theta')) && ~isempty(theta)
        renderOne(theta, 'theta', 'Rotation Angle', 'theta (deg)');
    end
end

function tf = isFixed(config, field)
    tf = false;
    if isfield(config,'use_manual_color_limits_map') && isfield(config.use_manual_color_limits_map, field)
        tf = logical(config.use_manual_color_limits_map.(field));
    elseif isfield(config,'use_manual_color_limits')
        tf = logical(config.use_manual_color_limits);
    end
end

function t = titleWithFrame(base, idx)
    if isempty(idx)
        t = base;
    else
        t = sprintf('%s - Frame %d', base, idx);
    end
end
