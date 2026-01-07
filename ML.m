function ML()
    % 配置
    % 训练集
    train_folder = './processed_data/'; 
    
    % 测试文件 
    test_mat_file = './test/chb01_03.mat';
    raw_edf_file  = './test/onset_labeled_chb01_03.edf';
    output_file   = 'svm_prediction.dat';
    
    win_size = 2560; 
    stride = 64;
    
    % 训练
    fprintf('准备训练数据\n');
    
    files = dir(fullfile(train_folder, '*.mat'));
    Train_X = []; % 特征矩阵 (N x 23)
    Train_Y = []; % 标签向量 (N x 1)
    
    for k = 1:length(files)
        fname = fullfile(files(k).folder, files(k).name);

        if contains(fname, 'chb01_03.mat'), continue; end
        
        try
            temp = load(fname, 'data');
            data_full = double(temp.data); % [Time x 24]
        catch
            continue; 
        end
        
        eeg = data_full(:, 1:23);
        labels = data_full(:, 24);
        
        % 提取发作样本
        seizure_indices = find(labels == 1);
        if isempty(seizure_indices), continue; end
        
        % 在发作区域滑动取样
        % 确定发作的起止点
        s_start = min(seizure_indices);
        s_end   = max(seizure_indices);
        
        pos_features = [];
        % 在发作段内滑动
        for s = s_start : stride : (s_end - win_size)
            seg = eeg(s : s+win_size-1, :);
            % 计算特征
            feat = calculate_features(seg); 
            pos_features = [pos_features; feat];
        end
        
        if isempty(pos_features), continue; end
        num_pos = size(pos_features, 1);
        
        % 提取正常样本
        non_seizure_indices = find(labels == 0);
        % 剔除边缘，防止越界
        valid_ns = non_seizure_indices(non_seizure_indices < (size(eeg,1) - win_size));
        
        if isempty(valid_ns)
            num_neg = 0;
        else
            % 随机抽样
            perm_idx = randperm(length(valid_ns), min(length(valid_ns), num_pos));
            sample_starts = valid_ns(perm_idx);
            
            neg_features = [];
            for j = 1:length(sample_starts)
                s = sample_starts(j);
                seg = eeg(s : s+win_size-1, :);
                feat = calculate_features(seg);
                neg_features = [neg_features; feat];
            end
        end
        
        % 汇总
        if ~isempty(pos_features) && ~isempty(neg_features)
            Train_X = [Train_X; pos_features; neg_features];
            Train_Y = [Train_Y; ones(size(pos_features,1),1); zeros(size(neg_features,1),1)];
            fprintf('已处理: %s (提取样本: 正%d / 负%d)\n', files(k).name, size(pos_features,1), size(neg_features,1));
        end
    end
    
    if isempty(Train_X)
        error('未提取到任何训练数据，检查路径或文件内容。');
    end
    
    % 训练
    fprintf('\n训练SVM模型\n');
    fprintf('总样本数: %d (特征维度: 23)\n', size(Train_X, 1));
    
    % 训练 SVM
    t_start = tic;
    svm_model = fitcsvm(Train_X, Train_Y, ...
        'KernelFunction', 'linear', ...
        'Standardize', true, ...
        'ClassNames', [0, 1]);
    
    svm_model = fitPosterior(svm_model); 
    
    fprintf('训练完成！耗时: %.2f 秒\n', toc(t_start));
    
    % 推理
    fprintf('\n对测试文件推理\n');
    fprintf('目标文件: %s\n', test_mat_file);
    
    % 读取测试数据
    load(test_mat_file, 'data');
    test_eeg = double(data(:, 1:23));
    [total_samples, ~] = size(test_eeg);
    
    % 目标对齐长度
    if ~isfile(raw_edf_file), error('找不到EDF文件'); end
    info = edfinfo(raw_edf_file);
    target_len = info.NumDataRecords * info.NumSamples(1);
    
    % 滑动窗口预测
    num_wins = floor((total_samples - win_size) / stride) + 1;
    pred_scores = zeros(num_wins, 1);
    pred_time = zeros(num_wins, 1);
    
    fprintf('正在提取测试集特征...\n');
    Test_Features = zeros(num_wins, 23);
    
    for i = 1:num_wins
        s = (i-1)*stride + 1;
        segment = test_eeg(s : s+win_size-1, :);
        Test_Features(i, :) = calculate_features(segment);
        pred_time(i) = s;
    end
    
    fprintf('正在 SVM 推理...\n');
    [~, score] = predict(svm_model, Test_Features);
    seizure_prob = score(:, 2);
    
    % 插值与保存
    fprintf('正在插值对齐...\n');
    
    x_target = (1:target_len)';
    
    % 插值概率曲线
    y_final = interp1(pred_time, seizure_prob, x_target, 'pchip', 'extrap');
    
    % 清洗
    y_final(y_final < 0) = 0;
    y_final(y_final > 1) = 1;
    y_final(isnan(y_final)) = 0;
    
    % 保存
    T = table(y_final, 'VariableNames', {'SVM_Probability'});
    if exist(output_file, 'file'), delete(output_file); end
    writetable(T, output_file, 'Delimiter', '\t');
    
    fprintf('=== 全部完成！ ===\n');
    fprintf('结果已保存: %s\n', output_file);
end

% 计算特征
function feat = calculate_features(segment)

    ll = sum(abs(diff(segment)));

    feat = log10(ll + 1); 
end