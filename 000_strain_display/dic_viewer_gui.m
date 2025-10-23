function dic_viewer_gui()
% dic_viewer_gui - Compact GUI to configure and run main_disp_strain_plot

    % Create UI
    f = uifigure('Name','DIC Results Viewer','Position',[100 100 540 520]);
    g = uigridlayout(f,[15 2]);
    g.RowHeight = repmat({'fit'},1,15);
    g.ColumnWidth = {'1x','1x'};

    % Image/MAT selectors
    uilabel(g,'Text','Image Folder:');
    hb1 = uigridlayout(g,[1 2]); hb1.ColumnWidth = {'1x', 80};
    edtImg = uieditfield(hb1,'text','Placeholder','Select image folder');
    uibutton(hb1,'Text','Browse', 'ButtonPushedFcn', @(~,~)browseFolder(edtImg));

    uilabel(g,'Text','Sequence MAT:');
    hb2 = uigridlayout(g,[1 2]); hb2.ColumnWidth = {'1x', 80};
    edtMat = uieditfield(hb2,'text','Placeholder','Select displacement_sequence.mat');
    uibutton(hb2,'Text','Browse', 'ButtonPushedFcn', @(~,~)browseFile(edtMat,{'*.mat','MAT files'}));

    % Overlay mode and frames
    uilabel(g,'Text','Overlay Mode:');
    ddOverlay = uidropdown(g,'Items',{'reference','deformed'},'Value','reference');

    uilabel(g,'Text','Frames (e.g., 1:10 or 1,5,8):');
    edtFrames = uieditfield(g,'text','Value','');

    % Fields to show
    uilabel(g,'Text','Displacements:');
    gpD = uigridlayout(g,[1 3]);
    cbU = uicheckbox(gpD,'Text','U','Value',true);
    cbV = uicheckbox(gpD,'Text','V','Value',true);
    cbMag = uicheckbox(gpD,'Text','Magnitude','Value',true);

    uilabel(g,'Text','Strains:');
    gpS = uigridlayout(g,[1 5]);
    cbExx = uicheckbox(gpS,'Text','exx','Value',true);
    cbEyy = uicheckbox(gpS,'Text','eyy','Value',true);
    cbExy = uicheckbox(gpS,'Text','exy','Value',true);
    cbEvm = uicheckbox(gpS,'Text','e\_vm','Value',true);
    cbTheta = uicheckbox(gpS,'Text','theta','Value',false);

    % Saving options
    cbSave = uicheckbox(g,'Text','Save figures','Value',true);
    cbOrg = uicheckbox(g,'Text','Organized folders','Value',true);
    uilabel(g,'Text','Output Dir:');
    hb3 = uigridlayout(g,[1 2]); hb3.ColumnWidth = {'1x', 80};
    edtOut = uieditfield(hb3,'text','Placeholder','Select output folder');
    uibutton(hb3,'Text','Browse', 'ButtonPushedFcn', @(~,~)browseFolder(edtOut));

    % Combined figure and video
    cbCombined = uicheckbox(g,'Text','Save combined figure','Value',false);
    cbVideo = uicheckbox(g,'Text','Export video from combined','Value',false);
    uilabel(g,'Text','Video FPS:');
    spFps = uispinner(g,'Limits',[1 60],'Value',5);
    uilabel(g,'Text','Video Path (.mp4):');
    hb4 = uigridlayout(g,[1 2]); hb4.ColumnWidth = {'1x', 80};
    edtVideo = uieditfield(hb4,'text','Placeholder','Path to video');
    uibutton(hb4,'Text','Browse', 'ButtonPushedFcn', @(~,~)browseSaveFile(edtVideo,{'*.mp4','MP4 video'}));

    % Color limit options per field
    uilabel(g,'Text','Fix Color Limits (per field):');
    glCL = uigridlayout(g,[9 3]);
    glCL.ColumnWidth = {90, 90, 90};
    % Header
    uilabel(glCL,'Text','Field','FontWeight','bold');
    uilabel(glCL,'Text','Min','FontWeight','bold');
    uilabel(glCL,'Text','Max','FontWeight','bold');
    % Rows for each field
    [cbFixU, spMinU, spMaxU] = rowCL(glCL,'U');
    [cbFixV, spMinV, spMaxV] = rowCL(glCL,'V');
    [cbFixM, spMinM, spMaxM] = rowCL(glCL,'Magnitude');
    [cbFixExx, spMinExx, spMaxExx] = rowCL(glCL,'exx');
    [cbFixEyy, spMinEyy, spMaxEyy] = rowCL(glCL,'eyy');
    [cbFixExy, spMinExy, spMaxExy] = rowCL(glCL,'exy');
    [cbFixEvm, spMinEvm, spMaxEvm] = rowCL(glCL,'e\_vm');
    [cbFixTheta, spMinTheta, spMaxTheta] = rowCL(glCL,'theta');

    % Speed / performance options
    cbShow = uicheckbox(g,'Text','Show figures (disable for fastest runs)','Value',true);
    cbFast = uicheckbox(g,'Text','Fast mode (no review, no pause)','Value',true);
    uilabel(g,'Text','Auto-advance delay (s):');
    spDelay = uispinner(g,'Limits',[0 10],'Value',0.1,'Step',0.1);
    cbParallel = uicheckbox(g,'Text','Use parallel (may incur startup overhead)','Value',false);

    % Interactions between speed options
    cbFast.ValueChangedFcn = @(src,~)onFastChanged(src, cbShow, spDelay);
    cbShow.ValueChangedFcn = @(src,~)onShowChanged(src, cbFast);

    % Run / Cancel
    btnRun = uibutton(g,'Text','Run','ButtonPushedFcn', @runClicked, 'BackgroundColor',[0.2 0.6 0.2], 'FontWeight','bold');
    btnCancel = uibutton(g,'Text','Close','ButtonPushedFcn', @(~,~)close(f));

    % Defaults
    try
        edtOut.Value = fullfile(pwd,'plots');
        def = localDefaultConfig();
        % Apply defaults to color-limit controls
        cbFixU.Value = def.use_manual_color_limits_map.U;     spMinU.Value = def.color_limits.U(1);        spMaxU.Value = def.color_limits.U(2);
        cbFixV.Value = def.use_manual_color_limits_map.V;     spMinV.Value = def.color_limits.V(1);        spMaxV.Value = def.color_limits.V(2);
        cbFixM.Value = def.use_manual_color_limits_map.Magnitude; spMinM.Value = def.color_limits.Magnitude(1); spMaxM.Value = def.color_limits.Magnitude(2);
        cbFixExx.Value = def.use_manual_color_limits_map.exx; spMinExx.Value = def.color_limits.exx(1);    spMaxExx.Value = def.color_limits.exx(2);
        cbFixEyy.Value = def.use_manual_color_limits_map.eyy; spMinEyy.Value = def.color_limits.eyy(1);    spMaxEyy.Value = def.color_limits.eyy(2);
        cbFixExy.Value = def.use_manual_color_limits_map.exy; spMinExy.Value = def.color_limits.exy(1);    spMaxExy.Value = def.color_limits.exy(2);
        cbFixEvm.Value = def.use_manual_color_limits_map.e_vm; spMinEvm.Value = def.color_limits.e_vm(1);  spMaxEvm.Value = def.color_limits.e_vm(2);
        if isfield(def.use_manual_color_limits_map,'theta'), cbFixTheta.Value = def.use_manual_color_limits_map.theta; end
        if isfield(def.color_limits,'theta')
            spMinTheta.Value = def.color_limits.theta(1); spMaxTheta.Value = def.color_limits.theta(2);
        else
            spMinTheta.Value = -5; spMaxTheta.Value = 5;
        end
    catch
    end

    function runClicked(~,~)
        cfg = struct();
        cfg.image_folder = strtrim(edtImg.Value);
        cfg.mat_file = strtrim(edtMat.Value);
        cfg.overlay_mode = ddOverlay.Value;
        cfg.frames = parseFrames(edtFrames.Value);
        cfg.fields_to_plot = struct();
        cfg.fields_to_plot.displacements = pick({'U',cbU.Value},{'V',cbV.Value},{'Magnitude',cbMag.Value});
        cfg.fields_to_plot.strains = pick({'exx',cbExx.Value},{'eyy',cbEyy.Value},{'exy',cbExy.Value},{'e_vm',cbEvm.Value},{'theta',cbTheta.Value});
        cfg.save_plots = cbSave.Value;
        cfg.organized_save = cbOrg.Value;
        cfg.output_dir = strtrim(edtOut.Value);
        cfg.make_combined_figure = cbCombined.Value;
        cfg.export_video = cbVideo.Value;
        cfg.video_fps = spFps.Value;
        cfg.video_path = strtrim(edtVideo.Value);
        % Performance toggles
        cfg.use_parallel = cbParallel.Value;
        % Speed controls
        if cbFast.Value
            cfg.show_figures = false;
            cfg.pause_between_frames = false;
            cfg.auto_advance = true;
            cfg.advance_delay = spDelay.Value;
        else
            cfg.show_figures = cbShow.Value;
            cfg.pause_between_frames = ~cbFast.Value && cbShow.Value; % pause only when showing and not fast
            cfg.auto_advance = ~cfg.pause_between_frames;
            cfg.advance_delay = spDelay.Value;
        end
        % Color limit controls
        cfg.use_manual_color_limits_map = struct( ...
            'U', cbFixU.Value, 'V', cbFixV.Value, 'Magnitude', cbFixM.Value, ...
            'exx', cbFixExx.Value, 'eyy', cbFixEyy.Value, 'exy', cbFixExy.Value, 'e_vm', cbFixEvm.Value);
        cfg.color_limits.U = [spMinU.Value, spMaxU.Value];
        cfg.color_limits.V = [spMinV.Value, spMaxV.Value];
        cfg.color_limits.Magnitude = [spMinM.Value, spMaxM.Value];
        cfg.color_limits.exx = [spMinExx.Value, spMaxExx.Value];
        cfg.color_limits.eyy = [spMinEyy.Value, spMaxEyy.Value];
        cfg.color_limits.exy = [spMinExy.Value, spMaxExy.Value];
        cfg.color_limits.e_vm = [spMinEvm.Value, spMaxEvm.Value];
        cfg.use_manual_color_limits_map.theta = cbFixTheta.Value;
        cfg.color_limits.theta = [spMinTheta.Value, spMaxTheta.Value];
        % Setup progress dialog
        pd = uiprogressdlg(f, 'Title','Processing', 'Message','Initializing...', 'Indeterminate','off', 'Value',0);
        drawnow;
        cfg.progress_callback = @(cur,total) progressUpdate(pd, cur, total);
        try
            run_disp_strain_plot(cfg);
        catch ME
            if isvalid(pd), close(pd); end
            uialert(f, ME.message, 'Error running');
            rethrow(ME);
        end
        if isvalid(pd), close(pd); end
    end
end

function [cbFix, spMin, spMax] = rowCL(parent, label)
    cbFix = uicheckbox(parent,'Text',label,'Value',false);
    spMin = uieditfield(parent,'numeric','Value',0);
    spMax = uieditfield(parent,'numeric','Value',1);
end


function browseFolder(edt)
    p = uigetdir(pwd,'Select folder');
    if isequal(p,0), return; end
    edt.Value = p;
end

function progressUpdate(pd, cur, total)
    if ~isvalid(pd), return; end
    if total <= 0
        pd.Value = 0; pd.Message = 'Preparing...';
    else
        pd.Value = max(0,min(1, cur/total));
        pd.Message = sprintf('Processing frame %d of %d', cur, total);
    end
    drawnow limitrate;
end

function onFastChanged(src, cbShow, spDelay)
    if src.Value
        cbShow.Value = false;
    end
    cbShow.Enable = matlab.lang.OnOffSwitchState(~src.Value);
    spDelay.Enable = 'on';
end

function onShowChanged(src, cbFast)
    if src.Value
        cbFast.Value = false;
    end
end

function browseFile(edt, filt)
    [f,p] = uigetfile(filt,'Select file');
    if isequal(f,0), return; end
    edt.Value = fullfile(p,f);
end

function browseSaveFile(edt, filt)
    [f,p] = uiputfile(filt,'Save file as');
    if isequal(f,0), return; end
    edt.Value = fullfile(p,f);
end

function frames = parseFrames(txt)
    txt = strtrim(string(txt));
    if txt == "", frames = []; return; end
    try
        if contains(txt, ':')
            frames = eval(['[', char(txt), ']']); %#ok<EVLC>
        else
            parts = regexp(char(txt), '[,\s]+', 'split');
            frames = str2double(parts);
        end
    catch
        frames = [];
    end
    frames = frames(~isnan(frames));
end

function out = pick(varargin)
    out = {};
    for k = 1:nargin
        name = varargin{k}{1}; val = varargin{k}{2};
        if val, out{end+1} = name; end %#ok<AGROW>
    end
    if isempty(out)
        out = {'none-selected'}; % avoid empty
    end
end

function s = localDefaultConfig()
    s.use_manual_color_limits_map = struct('U', false, 'V', false, 'Magnitude', false, ...
                                           'exx', false, 'eyy', false, 'exy', false, 'e_vm', false);
    s.color_limits = struct('U', [-5, 5], 'V', [-5, 5], 'Magnitude', [0, 10], ...
                            'exx', [-0.1, 0], 'eyy', [-0.1, 0], 'exy', [-0.1, 0.1], 'e_vm', [0, 0.3]);
end

function run_disp_strain_plot(config)
% Orchestrates loading data, computing strain, and plotting overlays.
    if ~isfield(config,'overlay_mode') || isempty(config.overlay_mode)
        config.overlay_mode = 'deformed';
    end
    if ~isfield(config,'colormap_name') || isempty(config.colormap_name)
        config.colormap_name = 'turbo';
    end
    if ~isfield(config,'overlay_alpha') || isempty(config.overlay_alpha)
        config.overlay_alpha = 0.6;
    end
    if ~isfield(config,'show_displacement_plots'), config.show_displacement_plots = true; end
    if ~isfield(config,'show_strain_plots'),       config.show_strain_plots = true;       end
    if ~isfield(config,'save_plots'),              config.save_plots = false;             end
    if ~isfield(config,'organized_save'),          config.organized_save = true;          end
    if ~isfield(config,'output_dir') || isempty(config.output_dir)
        config.output_dir = fullfile(pwd,'plots');
    end
    if ~isfield(config,'show_figures'),            config.show_figures = true;            end
    if ~isfield(config,'export_video'),            config.export_video = false;           end
    if ~isfield(config,'video_path'),              config.video_path = '';                end
    if ~isfield(config,'video_fps') || isempty(config.video_fps), config.video_fps = 5;  end
    if ~isfield(config,'use_manual_color_limits_map') || ~isfield(config,'color_limits')
        d = localDefaultConfig();
        if ~isfield(config,'use_manual_color_limits_map'), config.use_manual_color_limits_map = d.use_manual_color_limits_map; end
        if ~isfield(config,'color_limits'), config.color_limits = d.color_limits; end
    end
    if ~isfield(config,'strain_window_size') || isempty(config.strain_window_size)
        config.strain_window_size = 21;
    end
    if ~isfield(config,'use_parallel') || isempty(config.use_parallel)
        config.use_parallel = true;
    end
    if ~isfield(config,'downsample_step') || isempty(config.downsample_step)
        config.downsample_step = 3;
    end
    if ~isfield(config,'frames'), config.frames = []; end

    % Inputs
    img_folder = config.image_folder;
    if ~(ischar(img_folder) || isstring(img_folder)) || isempty(img_folder) || ~exist(img_folder,'dir')
        error('Image folder not set or not found.');
    end
    mat_path = config.mat_file;
    if ~(ischar(mat_path) || isstring(mat_path)) || isempty(mat_path) || ~exist(mat_path,'file')
        error('Result MAT file not set or not found.');
    end

    % Images list (common formats)
    exts = {'*.bmp','*.png','*.jpg','*.jpeg','*.tif','*.tiff'};
    img_list = [];
    for e = 1:numel(exts)
        img_list = [img_list; dir(fullfile(img_folder, exts{e}))]; %#ok<AGROW>
    end

    % Load DIC results
    s = load(mat_path);
    if isfield(s,'X_ref'), X_cells = s.X_ref; elseif isfield(s,'X'), X_cells = s.X; else, error('MAT missing X or X_ref'); end
    if isfield(s,'Y_ref'), Y_cells = s.Y_ref; elseif isfield(s,'Y'), Y_cells = s.Y; else, error('MAT missing Y or Y_ref'); end
    if ~isfield(s,'U') || ~isfield(s,'V'), error('MAT missing U and/or V'); end
    U_cells = s.U; V_cells = s.V;
    if ~iscell(U_cells) || ~iscell(V_cells) || ~iscell(X_cells) || ~iscell(Y_cells)
        error('U,V,X,Y must be cell arrays of size 1x(N-1).');
    end
    n = numel(U_cells);
    if n == 0 || n ~= numel(V_cells) || n ~= numel(X_cells) || n ~= numel(Y_cells)
        error('U,V,X,Y must have same non-zero length.');
    end

    % Frames
    if isempty(config.frames)
        frame_indices = 1:n;
    else
        frame_indices = config.frames(:)';
        frame_indices = frame_indices(frame_indices>=1 & frame_indices<=n);
        if isempty(frame_indices), frame_indices = 1:n; end
    end

    % Output dirs
    if config.save_plots && ~exist(config.output_dir,'dir')
        mkdir(config.output_dir);
    end

    % Video writer
    vw = [];
    vidW = []; vidH = [];
    if config.export_video
        if ~isfield(config,'video_path') || isempty(config.video_path)
            video_root = fullfile(config.output_dir, 'Results_figure');
            if ~exist(video_root,'dir'), mkdir(video_root); end
            config.video_path = fullfile(video_root, 'combined_video.mp4');
        else
            vr = fileparts(config.video_path);
            if ~isempty(vr) && ~exist(vr,'dir'), mkdir(vr); end
        end
        vw = VideoWriter(config.video_path, 'MPEG-4');
        vw.FrameRate = config.video_fps;
        open(vw);
    end

    total_frames = numel(frame_indices);
    if isfield(config,'progress_callback') && ~isempty(config.progress_callback)
        try feval(config.progress_callback, 0, total_frames); catch, end
    end

    % Reference background (image 1)
    reference_image_rgb = [];
    if ~isempty(img_list)
        try
            Iref = imread(fullfile(img_folder, img_list(1).name));
            if size(Iref,3) == 3, Iref = rgb2gray(Iref); end
            reference_image_rgb = cat(3, Iref, Iref, Iref);
        catch
            reference_image_rgb = [];
        end
    end

    use_deformed = strcmpi(config.overlay_mode,'deformed');
    for ii = 1:total_frames
        drawnow limitrate; % allow UI to refresh before heavy compute
        i = frame_indices(ii);
        X = X_cells{i}; Y = Y_cells{i}; U = U_cells{i}; V = V_cells{i};

        [exx, eyy, exy, theta] = calculate_strain_accelerated(X, Y, U, V, ...
            config.strain_window_size, config.use_parallel, config.downsample_step);
        e_avg = (exx + eyy) / 2;
        e_diff_rad = sqrt(((exx - eyy)/2).^2 + exy.^2);
        e1 = e_avg + e_diff_rad; e2 = e_avg - e_diff_rad;
        e_vm = (sqrt(2)/3) * sqrt((e1 - e2).^2 + e2.^2 + e1.^2);

        if use_deformed && ~isempty(img_list)
            % Align frame i with deformed image i+1
            img_idx = min(i+1, numel(img_list));
            Ibg = imread(fullfile(img_folder, img_list(img_idx).name));
            if size(Ibg,3) == 3, Ibg = rgb2gray(Ibg); end
            bg_rgb = cat(3, Ibg, Ibg, Ibg);
        else
            bg_rgb = reference_image_rgb;
        end

        if use_deformed
            if ~isempty(bg_rgb)
                [img_h, img_w, ~] = size(bg_rgb);
            else
                img_h = size(X,1); img_w = size(X,2);
            end
            [U_d, V_d, exx_d, eyy_d, exy_d, Xq, Yq, theta_d, ROI_d] = warp_fields_to_deformed(X, Y, U, V, exx, eyy, exy, img_h, img_w, theta);
            e_avg_d = (exx_d + eyy_d)/2;
            e_diff_rad_d = sqrt(((exx_d - eyy_d)/2).^2 + exy_d.^2);
            e1_d = e_avg_d + e_diff_rad_d; e2_d = e_avg_d - e_diff_rad_d;
            e_vm_d = (sqrt(2)/3) * sqrt((e1_d - e2_d).^2 + e2_d.^2 + e1_d.^2);

            if config.show_displacement_plots
                config.alpha_mask = ROI_d;
                [hU, namesU] = plot_displacements(Xq, Yq, U_d, V_d, bg_rgb, config, i, config.fields_to_plot.displacements);
                config.alpha_mask = [];
                if config.save_plots
                    if config.organized_save
                        saveFiguresOrganized(hU, namesU, fullfile(config.output_dir,'Results_figure'), 'Deformed image', i);
                    else
                        saveFigures(hU, fullfile(config.output_dir, sprintf('frame_%04d_disp_', i)));
                    end
                end
            end
            if config.show_strain_plots
                config.alpha_mask = ROI_d;
                [hS, namesS] = plot_strains(Xq, Yq, exx_d, eyy_d, exy_d, e_vm_d, bg_rgb, config, i, config.fields_to_plot.strains, theta_d);
                config.alpha_mask = [];
                if config.save_plots
                    if config.organized_save
                        saveFiguresOrganized(hS, namesS, fullfile(config.output_dir,'Results_figure'), 'Deformed image', i);
                    else
                        saveFigures(hS, fullfile(config.output_dir, sprintf('frame_%04d_strain_', i)));
                    end
                end
            end

            if isfield(config,'make_combined_figure') && config.make_combined_figure
                strains_struct = struct('exx',exx_d,'eyy',eyy_d,'exy',exy_d,'e_vm',e_vm_d);
                if exist('theta_d','var'), strains_struct.theta = theta_d; end
                config.alpha_mask = ROI_d;
                hC = plot_combined(Xq, Yq, bg_rgb, config, i, ...
                    struct('U',U_d,'V',V_d,'Magnitude',hypot(U_d,V_d)), ...
                    strains_struct);
                config.alpha_mask = [];
                if config.save_plots
                    out_dir = fullfile(config.output_dir,'Results_figure','Deformed image','combined');
                    if ~exist(out_dir,'dir'), mkdir(out_dir); end
                    out_path = fullfile(out_dir, sprintf('frame_%04d.png', i));
                    exportgraphics(hC, out_path, 'Resolution', 200);
                end
                if isfield(config,'auto_close_combined') && config.auto_close_combined
                    close(hC);
                end
            end
        else
            if config.show_displacement_plots
                [hU, namesU] = plot_displacements(X, Y, U, V, bg_rgb, config, i, config.fields_to_plot.displacements);
                if config.save_plots
                    if config.organized_save
                        saveFiguresOrganized(hU, namesU, fullfile(config.output_dir,'Results_figure'), 'Reference image', i);
                    else
                        saveFigures(hU, fullfile(config.output_dir, sprintf('frame_%04d_disp_', i)));
                    end
                end
            end
            if config.show_strain_plots
                [hS, namesS] = plot_strains(X, Y, exx, eyy, exy, e_vm, bg_rgb, config, i, config.fields_to_plot.strains, theta);
                if config.save_plots
                    if config.organized_save
                        saveFiguresOrganized(hS, namesS, fullfile(config.output_dir,'Results_figure'), 'Reference image', i);
                    else
                        saveFigures(hS, fullfile(config.output_dir, sprintf('frame_%04d_strain_', i)));
                    end
                end
            end
            if isfield(config,'make_combined_figure') && config.make_combined_figure
                strains_struct = struct('exx',exx,'eyy',eyy,'exy',exy,'e_vm',e_vm);
                if exist('theta','var'), strains_struct.theta = theta; end
                hC = plot_combined(X, Y, bg_rgb, config, i, ...
                    struct('U',U,'V',V,'Magnitude',hypot(U,V)), ...
                    strains_struct);
                if config.save_plots
                    out_dir = fullfile(config.output_dir,'Results_figure','Reference image','combined');
                    if ~exist(out_dir,'dir'), mkdir(out_dir); end
                    out_path = fullfile(out_dir, sprintf('frame_%04d.png', i));
                    exportgraphics(hC, out_path, 'Resolution', 200);
                end
                if isfield(config,'auto_close_combined') && config.auto_close_combined
                    close(hC);
                end
            end
        end

        if isfield(config,'progress_callback') && ~isempty(config.progress_callback)
            try feval(config.progress_callback, ii, total_frames); catch, end
        end
    end

    if ~isempty(vw)
        try close(vw); catch, end
        fprintf('Video saved: %s\n', config.video_path);
    end
    disp('Done.');
end

function saveFigures(figs, prefix)
    if isempty(figs), return; end
    for k = 1:numel(figs)
        if ~ishandle(figs(k)), continue; end
        try
            fname = sprintf('%s%02d.png', prefix, k);
            exportgraphics(figs(k), fname, 'Resolution', 200);
        catch
            saveas(figs(k), sprintf('%s%02d.png', prefix, k));
        end
    end
end

function saveFiguresOrganized(figs, names, root_dir, overlay_label, frame_idx)
    if isempty(figs), return; end
    if nargin < 5 || isempty(frame_idx), frame_idx = 1; end
    if nargin < 4 || isempty(overlay_label), overlay_label = 'Reference image'; end
    if nargin < 3 || isempty(root_dir), root_dir = fullfile(pwd, 'Results_figure'); end
    for k = 1:numel(figs)
        if ~ishandle(figs(k)), continue; end
        field_name = names{k};
        out_dir = fullfile(root_dir, overlay_label, field_name);
        if ~exist(out_dir, 'dir'), mkdir(out_dir); end
        out_path = fullfile(out_dir, sprintf('frame_%04d.png', frame_idx));
        try
            exportgraphics(figs(k), out_path, 'Resolution', 200);
        catch
            saveas(figs(k), out_path);
        end
    end
end

