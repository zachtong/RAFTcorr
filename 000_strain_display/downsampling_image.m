% Batch Image Downsampling with folder picker
% 功能: 选择输入/输出文件夹，批量按顺序缩小图片，文件名包含比例
% 作者: （你的名字）
% 用法: 直接运行，按提示选择文件夹与比例

clear; clc;

%% ---- 选择输入/输出文件夹 ----
inDir = uigetdir(pwd, '选择包含要处理图片的文件夹');
if isequal(inDir,0), disp('已取消。'); return; end

outDir = uigetdir(inDir, '选择处理后图片的保存文件夹');
if isequal(outDir,0), disp('已取消。'); return; end

%% ---- 输入缩放比例 ----
answer = inputdlg({'输入缩放比例 (0~1，例如 0.3 表示缩小到 30%):'}, ...
                  '设置缩放比例', 1, {'0.3'});
if isempty(answer), disp('已取消。'); return; end
scale = str2double(answer{1});
if ~(isnumeric(scale) && isfinite(scale) && scale>0 && scale<1)
    errordlg('缩放比例必须是 (0,1) 之间的小数，例如 0.3','参数错误');
    return;
end

%% ---- 收集待处理图片（多后缀）并排序 ----
exts = {'.jpg','.jpeg','.png','.tif','.tiff','.bmp'};
files = [];
for k = 1:numel(exts)
    files = [files; dir(fullfile(inDir, ['*' exts{k}]))]; %#ok<AGROW>
end
if isempty(files)
    errordlg('所选文件夹内未找到支持的图片格式。','无文件');
    return;
end

% 按文件名自然排序（简单字母序；如需更严格的自然数值排序可自行替换）
[~, idx] = sort(lower({files.name}));
files = files(idx);

%% ---- 处理并保存 ----
wb = waitbar(0, '正在处理图片，请稍候...');
cleanup = onCleanup(@() (ishandle(wb) && close(wb)));

scaleTag = ['_s' strrep(num2str(scale),'.','p')]; % 例如 0.3 -> _s0p3

numOK = 0; numFail = 0;
for i = 1:numel(files)
    try
        waitbar(i/numel(files), wb, sprintf('处理中: %s (%d/%d)', ...
            files(i).name, i, numel(files)));

        inPath = fullfile(files(i).folder, files(i).name);
        I = imread(inPath);

        % 缩放（使用默认双三次插值）
        I_down = imresize(I, scale);

        % 生成输出文件名：原名 + 比例标签 + 原扩展名
        [~, base, ext] = fileparts(files(i).name);
        outPath = fullfile(outDir, [base, scaleTag, ext]);

        % 保存（保持原扩展名；若为JPEG可根据需要调质量）
        if any(strcmpi(ext, {'.jpg','.jpeg'}))
            imwrite(I_down, outPath, 'Quality', 95);
        else
            imwrite(I_down, outPath);
        end

        numOK = numOK + 1;
    catch ME
        warning('处理失败：%s\n原因：%s', files(i).name, ME.message);
        numFail = numFail + 1;
    end
end

if ishandle(wb), close(wb); end
fprintf('完成！成功: %d, 失败: %d\n输出位置：%s\n', numOK, numFail, outDir);
