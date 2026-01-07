function AIC()
    % 配置
    raw_edf_file = './test/onset_labeled_chb01_03.edf';
    input_mat_file = './test/chb01_03.mat'; 
    
    max_order = 50;
    win_len = 2560; 

    fprintf('AR模型最佳阶数扫描(基于AIC准则)\n');
    
    if ~isfile(input_mat_file), error('找不到预处理文件'); end
    load(input_mat_file, 'data');
    labels = data(:, 24);
    eeg = double(data(:, 1:23));
    
    % 提取发作数据
    seizure_idxs = find(labels == 1);
    if isempty(seizure_idxs)
        warning('未找到发作标记，将使用一段背景数据进行分析...');
        start_idx = 1000;
    else
        mid_ptr = round(length(seizure_idxs)/2);
        center_idx = seizure_idxs(mid_ptr);
        start_idx = center_idx; 
        fprintf('已定位发作中心: %d\n', center_idx);
    end
    
    % 截取
    % 建议取方差最大的那个通道（发作最明显的）
    segment = eeg(start_idx : start_idx+win_len-1, :);
    [~, ch_idx] = max(var(segment)); 
    x = segment(:, ch_idx);
    x = x - mean(x); % 去直流
    
    N = length(x);
    aic_values = zeros(max_order, 1);
    
    fprintf('正在扫描 1-%d 阶 (基于通道 %d)...\n', max_order, ch_idx);
    
    % 2. 循环计算 AIC
    for p = 1:max_order
        % aryule 返回: a(系数), e(噪声方差)
        [~, e] = aryule(x, p);
        
        % AIC 公式
        % log(e) 代表误差大小
        % 2*p/N 代表对复杂度的惩罚
        aic = log(e) + 2*p/N;
        
        aic_values(p) = aic;
    end
    
    % 3. 找最小值
    [min_aic, best_p] = min(aic_values);
    
    % 4. 绘图
    figure('Color', 'w', 'Name', 'Best AR Order Selection');
    plot(1:max_order, aic_values, 'b-o', 'LineWidth', 1.5);
    hold on;
    plot(best_p, min_aic, 'rp', 'MarkerSize', 12, 'MarkerFaceColor', 'r');
    grid on;
    xlabel('AR Model Order (p)');
    ylabel('AIC Value (Lower is Better)');
    title(sprintf('最佳阶数 AIC 扫描'));
    
    fprintf('------------------------------------------------\n');
    fprintf('扫描完成！\n');
    fprintf('>> 理论最佳阶数: %d\n', best_p);
end