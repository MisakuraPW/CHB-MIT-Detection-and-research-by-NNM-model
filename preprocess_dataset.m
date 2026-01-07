function preprocess_dataset()
    % 配置
    dataset_path = "./data/chb01/onset_labeled/"; 
    output_path = "./processed_data/";
        
    if ~exist(output_path, 'dir')
        mkdir(output_path);
    end

    % 通道
    target_channels = [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", ...
        "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2", ...
        "FP2-F8", "F8-T8", "T8-P8", "P8-O2", "FZ-CZ", "CZ-PZ", ...
        "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8"
    ];

    % 滤波器
    Fs = 256; 
    [b, a] = butter(4, [0.5, 40] / (Fs/2), 'bandpass');

    files = dir(fullfile(dataset_path, "*.edf"));
    
    disp("开始预处理……");

    for i = 1:length(files)
        file_name = files(i).name;
        full_path = fullfile(files(i).folder, file_name);
        
        % 生成输出文件名
        [~, name_core, ~] = fileparts(file_name);
        name_core = replace(name_core, 'onset_labeled_', '');
        save_file_name = fullfile(output_path, name_core + ".mat");
        
        if exist(save_file_name, 'file')
            fprintf("文件 (%d/%d): %s 已存在 -> 跳过\n", i, length(files), save_file_name);
            continue;
        end
        
        fprintf("处理文件 (%d/%d): %s ... ", i, length(files), file_name);
        
        try
            % 读取数字
            data_tbl = edfread(full_path);
            info = edfinfo(full_path);
            original_labels = string(info.SignalLabels);
            
            % 读取标注
            try
                [~, annotations] = edfread(full_path, 'SelectedDataTypes', 'Annotations');
            catch
                 if isprop(data_tbl, 'Properties') && ...
                    isprop(data_tbl.Properties, 'CustomProperties') && ...
                    isfield(data_tbl.Properties.CustomProperties, 'Annotations')
                     annotations = data_tbl.Properties.CustomProperties.Annotations;
                 else
                     temp_info = edfinfo(full_path);
                     annotations = temp_info.Annotations;
                 end
            end
            
            % 通道提取
            selected_data = [];
            valid_file = true;
            
            for ch = target_channels
                idx = find(strcmpi(original_labels, ch));
                if isempty(idx)
                    fprintf(" [跳过] 缺失通道: %s\n", ch);
                    valid_file = false;
                    break; 
                end
                
                raw_col = data_tbl{:, idx(1)};
                if iscell(raw_col), raw_col = cell2mat(raw_col); end
                selected_data = [selected_data, double(raw_col)];
            end
            
            if ~valid_file, continue; end
            
            % 滤波
            filtered_data = filtfilt(b, a, selected_data);
            
            num_samples = size(filtered_data, 1);
            label_vector = zeros(num_samples, 1);
            
            if ~isempty(annotations) && height(annotations) > 0
                for k = 1:height(annotations)
                    if iscell(annotations.Annotations)
                        anno_str = string(annotations.Annotations{k});
                    else
                        anno_str = string(annotations.Annotations(k));
                    end
                    
                    if contains(anno_str, "seizure", 'IgnoreCase', true)
                        onset = seconds(annotations.Onset(k));
                        duration = seconds(annotations.Duration(k));
                        start_idx = max(1, round(onset * Fs));
                        end_idx = min(num_samples, round((onset + duration) * Fs));
                        
                        label_vector(start_idx:end_idx) = 1; 
                    end
                end
            end
            
            % 拼接
            data = [filtered_data, label_vector];
            
            save(save_file_name, 'data');
            fprintf("保存成功! 矩阵大小: [%d x %d]\n", size(data, 1), size(data, 2));
            
            clear data filtered_data selected_data data_tbl
            
        catch ME
            fprintf(" [出错] %s\n", ME.message);
        end
    end
    fprintf("=== 全部预处理结束 ===\n");
end