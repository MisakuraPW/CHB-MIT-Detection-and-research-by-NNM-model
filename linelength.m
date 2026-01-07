function linelength()
    % 配置
    raw_edf_file = './test/onset_labeled_chb01_03.edf';
    input_mat_file = './test/chb01_03.mat'; 
    output_file = 'line_length.dat';
    
    window_size = 2560; 
    stride = 64;

    fprintf('生成特征\n');
    
    % 获取目标长度
    if ~isfile(raw_edf_file), error('找不到EDF文件'); end
    info = edfinfo(raw_edf_file);
    target_len = info.NumDataRecords * info.NumSamples;
    
    % 加载数据
    load(input_mat_file, 'data'); 
    eeg = double(data(:, 1:23));
    [mat_len, ~] = size(eeg);
    
    num_wins = floor((mat_len - window_size) / stride) + 1;
    sparse_vals = zeros(num_wins, 1);
    sparse_time = zeros(num_wins, 1);
    
    fprintf('正在计算...\n');
    
    parfor i = 1:num_wins
        s = (i-1)*stride + 1;
        segment = eeg(s : s+window_size-1, :);
        
        ch_ll = zeros(23, 1);
        for ch = 1:23
            sig = segment(:, ch);
            
            ll = sum(abs(diff(sig)));

            ch_ll(ch) = log10(ll + 1); 
        end
        
        sparse_vals(i) = mean(ch_ll); 
        sparse_time(i) = s;
    end
    
    % 插值对齐
    x_target = (1:target_len)';
    feature_col = interp1(sparse_time, sparse_vals, x_target, 'pchip', 'extrap');
    
    % 清洗
    feature_col(isnan(feature_col)) = 0;
    
    % 保存
    T = table(feature_col, 'VariableNames', {'Line_Length'});
    if exist(output_file, 'file'), delete(output_file); end
    writetable(T, output_file, 'Delimiter', '\t');
    
    fprintf('完成！已生成: %s\n', output_file);
end