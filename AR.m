function AR()
    % 配置
    raw_edf_file = './test/onset_labeled_chb01_03.edf';
    input_mat_file = './test/chb01_03.mat'; 
    output_file = 'ar_feature_multi.dat';
    
    window_size = 2560; 
    stride = 64;      
    ar_order = 20;
    
    save_all_channels = true; 
    
    if ~isfile(raw_edf_file), error('找不到EDF文件'); end
    info = edfinfo(raw_edf_file);
    
    target_len = info.NumDataRecords * info.NumSamples(1);

    load(input_mat_file, 'data'); 
    eeg = double(data(:, 1:23));
    [mat_len, num_channels] = size(eeg);
    
    num_wins = floor((mat_len - window_size) / stride) + 1;
    
    sparse_vals = zeros(num_wins, 3 + num_channels); 
    sparse_time = zeros(num_wins, 1);
    
    fprintf('正在计算...\n');

    for i = 1:num_wins
        s = (i-1)*stride + 1;
        segment = eeg(s : s+window_size-1, :);
        
        win_npes = zeros(num_channels, 1);
        win_vars = zeros(num_channels, 1);
        
        for ch = 1:num_channels
            x = segment(:, ch);
            x = x - mean(x);
            v = var(x);
            win_vars(ch) = v; % 方差
            
            if v < 1e-8
                win_npes(ch) = 1.0;
            else
                [~, noise] = aryule(x, ar_order);
                npe = noise / v;
                win_npes(ch) = npe;
            end
        end
        
        % Min NPE
        val_min = min(win_npes);
        
        % Mean NPE
        val_mean = mean(win_npes);
        
        % Max Variance Channel NPE
        [~, max_var_idx] = max(win_vars);
        val_max_var = win_npes(max_var_idx);
        
        % 存储
        sparse_vals(i, :) = [val_min, val_mean, val_max_var, win_npes'];
        sparse_time(i) = s;
    end
    
    fprintf('计算完成，正在插值对齐...\n');
    
    % 插值对齐
    x_target = (1:target_len)';
    feature_matrix = zeros(target_len, size(sparse_vals, 2));
    for col = 1:size(sparse_vals, 2)
        y_col = interp1(sparse_time, sparse_vals(:, col), x_target, 'pchip', 'extrap');
        
        % 清洗
        y_col(isnan(y_col)) = 1; 
        y_col(isinf(y_col)) = 1;
        
        feature_matrix(:, col) = y_col;
    end
    
    % 保存
    if save_all_channels
        % 构造 26 个变量名
        var_names = {'AR-Min', 'AR-Mean', 'AR-MaxVar-Ch'};
        for k = 1:num_channels
            var_names{end+1} = sprintf('Ch-%d', k);
        end
        T = array2table(feature_matrix, 'VariableNames', var_names);
    else
        T = array2table(feature_matrix(:, 1:3), 'VariableNames', {'AR-Min', 'AR-Mean', 'AR-MaxVar-Ch'});
    end
    
    if exist(output_file, 'file'), delete(output_file); end
    writetable(T, output_file, 'Delimiter', '\t');
    
    fprintf('完成！已生成: %s\n', output_file);
end