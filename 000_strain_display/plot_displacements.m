function [figs, names] = plot_displacements(X, Y, U, V, ref_rgb, config, frame_index, fields)
% plot_displacements
% Render U, V, and |U,V| as heatmaps aligned to the provided grid using
% imagesc with AlphaData to respect ROI/holes. Works for both reference and
% deformed domains depending on X,Y.

    if nargin < 7, frame_index = []; end
    if nargin < 8 || isempty(fields), fields = {'U','V','Magnitude'}; end
    cm = config.colormap_name; if isempty(cm), cm = 'parula'; end

    figs = gobjects(0);
    names = {};
    vis = 'on'; if isfield(config,'show_figures') && ~config.show_figures, vis = 'off'; end

    % Compute plotting bounds from X,Y (pixel centers). Use +/-0.5 for edges.
    finiteXY = isfinite(X) & isfinite(Y);
    if any(finiteXY(:))
        xmin = min(X(finiteXY)); xmax = max(X(finiteXY));
        ymin = min(Y(finiteXY)); ymax = max(Y(finiteXY));
    else
        [m,n] = size(U);
        xmin = 0.5; xmax = n+0.5; ymin = 0.5; ymax = m+0.5;
    end
    xdata = [xmin-0.5, xmax+0.5];
    ydata = [ymin-0.5, ymax+0.5];

    % Prepare magnitude
    mag = hypot(U, V);

    % Helper to render one field via imagesc with ROI-based alpha
    function renderOne(Z, fieldName, titleText, cbarLabel)
        figs(end+1) = figure('Name', titleWithFrame(titleText, frame_index), 'Visible', vis); %#ok<AGROW>
        ax = gca; %#ok<NASGU>
        if ~isempty(ref_rgb), imshow(ref_rgb, 'InitialMagnification','fit'); hold on; end
        ROI = isfinite(Z) & finiteXY; % respect holes/NaNs
        if isfield(config,'alpha_mask') && ~isempty(config.alpha_mask) && all(size(config.alpha_mask) == size(Z))
            ROI = ROI & logical(config.alpha_mask);
        end
        h = imagesc(xdata, ydata, Z);
        set(h, 'AlphaData', double(ROI) * config.overlay_alpha);
        try colormap(cm); catch, colormap('parula'); end
        if isFixed(config, fieldName), caxis(config.color_limits.(fieldName)); end
        cb = colorbar; ylabel(cb, cbarLabel);
        axis image; set(gca,'YDir','reverse'); title(titleWithFrame(titleText, frame_index)); hold off;
        names{end+1} = fieldName; %#ok<AGROW>
    end

    if any(strcmpi(fields,'U'))
        renderOne(U, 'U', 'U Displacement', 'U (pixels)');
    end
    if any(strcmpi(fields,'V'))
        renderOne(V, 'V', 'V Displacement', 'V (pixels)');
    end
    if any(strcmpi(fields,'Magnitude'))
        renderOne(mag, 'Magnitude', 'Magnitude', 'Magnitude (pixels)');
    end
end

function t = titleWithFrame(base, idx)
    if isempty(idx)
        t = base;
    else
        t = sprintf('%s - Frame %d', base, idx);
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
