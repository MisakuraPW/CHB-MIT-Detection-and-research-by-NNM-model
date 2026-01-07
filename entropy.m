function entropy()
    % 配置
    raw_edf_file = './test/onset_labeled_chb01_03.edf';
    input_mat_file = './test/chb01_03.mat'; 
    output_file = 'entropy.dat';
    
    window_size = 2560;
    stride = 64;

    fprintf('熵特征\n');
 
    if ~isfile(raw_edf_file), error('找不到EDF文件'); end
    info = edfinfo(raw_edf_file);
    target_len = info.NumDataRecords * info.NumSamples(1);
    fprintf('目标对齐长度: %d 点\n', target_len);
    
    load(input_mat_file, 'data'); 
    eeg = double(data(:, 1:23));
    [mat_len, num_channels] = size(eeg);
    
    num_wins = floor((mat_len - window_size) / stride) + 1;
    
    sparse_vals = zeros(num_wins, 9); 
    sparse_time = zeros(num_wins, 1);
    
    h_win = hann(window_size);
    
    fprintf('正在计算...\n');
    
    for i = 1:num_wins
        s = (i-1)*stride + 1;
        segment = eeg(s : s+window_size-1, :);
        
        ch_shannon = zeros(num_channels, 1);
        ch_spectral = zeros(num_channels, 1);
        ch_sample = zeros(num_channels, 1);
        ch_vars = zeros(num_channels, 1);
        
        for ch = 1:num_channels
            sig = segment(:, ch);
            
            % 预处理
            sig = sig - mean(sig);
            ch_vars(ch) = var(sig); 
            
            % 香农熵
            p = histcounts(sig, 50, 'Normalization', 'probability');
            p(p==0) = [];
            ch_shannon(ch) = -sum(p .* log2(p));
            
            % 频域熵
            sig_win = sig .* h_win;
            P2 = abs(fft(sig_win)/window_size);
            P1 = P2(1:floor(window_size/2)+1);
            P1(2:end-1) = 2*P1(2:end-1);
            P_norm = P1 / (sum(P1) + 1e-12);
            ch_spectral(ch) = -sum(P_norm .* log2(P_norm + 1e-12)) / log2(length(P_norm));
            
            % 样本熵
            sig_down = sig(1:4:end); 
            ch_sample(ch) = calc_sampen_fast(sig_down, 2, 0.2);
        end
        
        % 聚合
        [~, max_var_idx] = max(ch_vars);
        
        shan_mv = ch_shannon(max_var_idx);
        shan_mean = mean(ch_shannon);
        shan_max = max(ch_shannon);
        
        spec_mv = ch_spectral(max_var_idx);
        spec_mean = mean(ch_spectral);
        spec_min = min(ch_spectral);
        
        samp_mv = ch_sample(max_var_idx);
        samp_mean = mean(ch_sample);
        samp_min = min(ch_sample);
        
        sparse_vals(i, :) = [shan_mv, shan_mean, shan_max, ...
                             spec_mv, spec_mean, spec_min, ...
                             samp_mv, samp_mean, samp_min];
        sparse_time(i) = s;
    end
    
    fprintf('计算完成，正在插值对齐...\n');
    
    x_target = (1:target_len)';
    feature_matrix = zeros(target_len, 9);
    
    % 循环插值每一列
    for col = 1:9
        y_col = interp1(sparse_time, sparse_vals(:, col), x_target, 'pchip', 'extrap');
        
        % 清洗 NaN / Inf
        y_col(isnan(y_col)) = 0; 
        y_col(isinf(y_col)) = 0;
        
        feature_matrix(:, col) = y_col;
    end
    
    % 保存
    var_names = { ...
        'Shannon-MV', 'Shannon-Mean', 'Shannon-Max', ...
        'Spectral-MV', 'Spectral-Mean', 'Spectral-Min', ...
        'Sample-MV',  'Sample-Mean',  'Sample-Min'};
        
    T = array2table(feature_matrix, 'VariableNames', var_names);
    
    if exist(output_file, 'file'), delete(output_file); end
    writetable(T, output_file, 'Delimiter', '\t');
    
    fprintf('文件已生成: %s\n', output_file);
end

function saen = calc_sampen_fast(x, m, r_factor)
    N = length(x);
    r = r_factor * std(x);
    if r < 1e-10, saen=0; return; end

    Xm = zeros(N-m, m);
    for k=1:m, Xm(:,k) = x(k:N-m+k-1); end

    B = 0; A = 0;

    template_indices = 1:3:(N-m); 
    
    for i = template_indices

        temp = Xm(i, :);
        % 计算该模板与其他所有向量的距离
        dists = max(abs(Xm - temp), [], 2);
        % 统计小于 r 的个数
        count = sum(dists < r) - 1;
        B = B + count;
        % 统计 m+1 维匹配
        % 找出 m 维匹配的索引
        match_idxs = find(dists < r);
        match_idxs(match_idxs == i) = [];
        % 检查 m+1 位
        valid_mask = (i + m <= N) & (match_idxs + m <= N);
        valid_matches = match_idxs(valid_mask);
        
        if (i + m <= N) && ~isempty(valid_matches)
             diffs = abs(x(i+m) - x(valid_matches+m));
             A = A + sum(diffs < r);
        end
    end
    
    if A==0 || B==0, saen=0; else, saen = -log(A/B); end
end