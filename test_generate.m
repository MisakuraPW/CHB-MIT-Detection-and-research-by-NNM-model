function test_generate()
    % ================= 配置 =================
    raw_edf = './data/chb01/onset_labeled/onset_labeled_chb01_03.edf';
    pre_mat = './processed_data/chb01_03.mat';
    
    out_dir = './test_dataset/';
    if ~exist(out_dir, 'dir'), mkdir(out_dir); end
    
    % 截取长度：发作中心前后各 300秒 (共10分钟)
    half_sec = 300; 
    Fs = 256;
    % =======================================
    
    fprintf('=== 1. 正在定位发作中心 ===\n');
    if ~isfile(pre_mat), error('找不到预处理文件: %s', pre_mat); end
    load(pre_mat, 'data'); 
    labels = data(:, 24);
    
    seizure_idx = find(labels == 1);
    if isempty(seizure_idx), error('未找到发作标记'); end
    center = round(mean(seizure_idx));
    
    s_idx = max(1, center - half_sec*Fs);
    e_idx = min(length(labels), center + half_sec*Fs);
    
    fprintf('锁定范围: %d -> %d (时长 %.1f 秒)\n', s_idx, e_idx, (e_idx-s_idx)/Fs);

    % === 2. 生成模型切片 (.mat) ===
    fprintf('生成 test_sample.mat (给AR/熵/NN使用)...\n');
    data = data(s_idx:e_idx, :); 
    save(fullfile(out_dir, 'test_sample.mat'), 'data');

    % === 3. 生成可视化切片 (.edf) ===
    fprintf('生成 test_sample.edf (给App显示)...\n');
    warning('off', 'signal:edfread:NonUniqueSignalLabels');
    
    info = edfinfo(raw_edf);
    
    fprintf('正在读取原始 EDF 文件...\n');
    [full_tbl, full_annots] = edfread(raw_edf);
    
    % 展开 Block 格式数据
    if height(full_tbl) < s_idx
        fprintf('正在展开 Block 数据...\n');
        temp_data = full_tbl{:, :}; 
        if iscell(temp_data)
            num_channels = size(temp_data, 2);
            expanded_data = [];
            for c = 1:num_channels
                col_data = vertcat(temp_data{:, c});
                expanded_data = [expanded_data, col_data];
            end
            raw_vals_full = expanded_data;
        else
            raw_vals_full = temp_data;
        end
    else
        raw_vals_full = full_tbl{:, :};
    end
    
    % 安全截取
    total_samples_real = size(raw_vals_full, 1);
    real_e_idx = min(e_idx, total_samples_real);
    
    % [核心修复 1] 去掉转置！保持 [Samples x Channels]
    raw_vals = double(raw_vals_full(s_idx:real_e_idx, :)); 
    
    fprintf('截取数据维度: %d 行 x %d 列 (应为 Sample x Channel)\n', size(raw_vals,1), size(raw_vals,2));
    
    % 构造新头文件
    hdr = edfheader("EDF");
    hdr.NumSignals = info.NumSignals;
    
    % [核心修复 2] 强制标签唯一，防止重复通道名报错
    hdr.SignalLabels = matlab.lang.makeUniqueStrings(info.SignalLabels);
    
    % [核心修复 3] 强制转换为 double，防止类型不匹配
    hdr.PhysicalDimensions = string(info.PhysicalDimensions); % 单位通常是字符串
    hdr.PhysicalMin = double(info.PhysicalMin);
    hdr.PhysicalMax = double(info.PhysicalMax);
    hdr.DigitalMin = double(info.DigitalMin);
    hdr.DigitalMax = double(info.DigitalMax);
    hdr.TransducerTypes = string(info.TransducerTypes);
    hdr.Prefilter = string(info.Prefilter);
    hdr.Patient = string(info.Patient);
    hdr.Recording = "Clip chb01_03";
    hdr.SampleRate = double(info.NumSamples) ./ seconds(info.DataRecordDuration);
    
    % 修正标注时间
    slice_start_sec = (s_idx-1)/Fs;
    slice_end_sec = real_e_idx/Fs;
    
    new_on = []; 
    new_dur = []; 
    new_txt = []; 
    
    if ~isempty(full_annots)
        for k=1:height(full_annots)
            orig_on_sec = seconds(full_annots.Onset(k));
            if (orig_on_sec < slice_end_sec) && (orig_on_sec + seconds(full_annots.Duration(k)) > slice_start_sec)
                new_on = [new_on; max(0, orig_on_sec - slice_start_sec)];
                new_dur = [new_dur; full_annots.Duration(k)];
                new_txt = [new_txt; string(full_annots.Annotations{k})];
            end
        end
    end
    
    out_edf = fullfile(out_dir, 'test_sample.edf');
    
    if ~isempty(new_on)
        % 修正 timetable 创建
        annot = timetable(seconds(new_on), new_txt, new_dur, 'VariableNames', {'Annotations','Duration'});
        edfwrite(out_edf, hdr, raw_vals, annot);
    else
        edfwrite(out_edf, hdr, raw_vals);
    end
    
    warning('on', 'signal:edfread:NonUniqueSignalLabels');
    fprintf('=== 完成！请使用 test_dataset 里的文件 ===\n');
end