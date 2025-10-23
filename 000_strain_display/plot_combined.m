function hFig = plot_combined(X, Y, bg_rgb, config, frame_index, disp_fields, strain_fields)
% plot_combined
% Build a multi-panel figure with selected fields using imagesc + AlphaData
% for robust alignment on both reference and deformed domains.

    % Collect selected field names in order
    dnames = config.fields_to_plot.displacements(:)';
    snames = config.fields_to_plot.strains(:)';
    fields = [dnames, snames];
    N = numel(fields);
    ncols = min(3, max(1, N));
    nrows = ceil(N / ncols);

    vis = 'on'; if isfield(config,'show_figures') && ~config.show_figures, vis = 'off'; end
    hFig = figure('Name', sprintf('Combined - Frame %d', frame_index), 'Color','w', 'Visible', vis);
    tiledlayout(hFig, nrows, ncols, 'Padding','compact', 'TileSpacing','compact');

    cm = config.colormap_name; if isempty(cm), cm = 'parula'; end

    finiteXY = isfinite(X) & isfinite(Y);
    if any(finiteXY(:))
        xmin = min(X(finiteXY)); xmax = max(X(finiteXY));
        ymin = min(Y(finiteXY)); ymax = max(Y(finiteXY));
    else
        [m,n] = size(X);
        xmin = 0.5; xmax = n+0.5; ymin = 0.5; ymax = m+0.5;
    end
    xdata = [xmin-0.5, xmax+0.5];
    ydata = [ymin-0.5, ymax+0.5];

    for idx = 1:N
        nexttile;
        if ~isempty(bg_rgb), imshow(bg_rgb, 'InitialMagnification','fit'); hold on; end
        fname = fields{idx};
        if ismember(fname, {'U','V','Magnitude'})
            Z = disp_fields.(fname);
        else
            Z = strain_fields.(fname);
        end
        ROI = isfinite(Z) & finiteXY;
        if isfield(config,'alpha_mask') && ~isempty(config.alpha_mask) && all(size(config.alpha_mask) == size(Z))
            ROI = ROI & logical(config.alpha_mask);
        end
        h = imagesc(xdata, ydata, Z);
        set(h, 'AlphaData', double(ROI) * config.overlay_alpha);
        try colormap(cm); catch, colormap('parula'); end
        if isFixed(config, fname) && isfield(config.color_limits, fname)
            caxis(config.color_limits.(fname));
        end
        title(sprintf('%s - F%d', fname, frame_index)); axis image; set(gca,'YDir','reverse'); hold off;
        colorbar;
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
