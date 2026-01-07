function viewEEG()
    % 配置
    input_file = 'nmm_data.mat';
    
    target_k_low = 10;  
    target_k_high = 90; 
    
    plot_duration = 5; 
    
    % 加载数据
    if ~isfile(input_file)
        error('找不到文件 %s。', input_file);
    end
    fprintf('正在加载 %s ...\n', input_file);
    load(input_file, 'generated_data', 'K_values', 'fs');
    
    [~, idx_low] = min(abs(K_values - target_k_low));
    [~, idx_high] = min(abs(K_values - target_k_high));
    
    real_k_low = K_values(idx_low);
    real_k_high = K_values(idx_high);
    
    fprintf('对比展示:\n 1. 正常态 K = %d\n 2. 发作态 K = %d\n', real_k_low, real_k_high);
    
    sig_low = squeeze(generated_data(idx_low, 1:3, :));
    sig_high = squeeze(generated_data(idx_high, 1:3, :));
    
    points_to_plot = fs * plot_duration;
    t = (0:points_to_plot-1) / fs;
    
    sig_low_plot = sig_low(:, 1:points_to_plot);
    sig_high_plot = sig_high(:, 1:points_to_plot);
    
    % 时域波形对比
    figure('Color', 'w', 'Name', 'NMM Time Domain Comparison', 'Position', [100, 100, 1000, 600]);
    
    subplot(2, 1, 1);
    offset = 5;
    hold on;
    for ch = 1:3
        plot(t, (sig_low_plot(ch, :) - mean(sig_low_plot(ch, :))) + (ch-1)*offset, ...
             'LineWidth', 1.2);
    end
    title(sprintf('正常背景活动 (K=%d)', real_k_low), 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Amplitude (mV)');
    xlabel('Time (s)');
    legend('Channel 1', 'Channel 2', 'Channel 3', 'Location', 'northeast');
    grid on;
    set(gca, 'FontSize', 12);
    ylim([-5, 15]);
    
    subplot(2, 1, 2);
    offset_high = 20; % 发作时幅值大，偏移量要大一点
    hold on;
    for ch = 1:3
        plot(t, (sig_high_plot(ch, :) - mean(sig_high_plot(ch, :))) + (ch-1)*offset_high, ...
             'LineWidth', 1.2);
    end
    title(sprintf('癫痫发作活动 (K=%d)', real_k_high), 'FontSize', 14, 'FontWeight', 'bold', 'Color', 'r');
    ylabel('Amplitude (mV)');
    xlabel('Time (s)');
    grid on;
    set(gca, 'FontSize', 12);
    
    % 频域功率谱对
    figure('Color', 'w', 'Name', 'NMM PSD Comparison', 'Position', [150, 150, 800, 500]);
    
    hold on;
    [pxx_low, f] = pwelch(sig_low(1, :) - mean(sig_low(1, :)), fs, [], [], fs);
    plot(f, 10*log10(pxx_low), 'b-', 'LineWidth', 2, 'DisplayName', sprintf('Normal (K=%d)', real_k_low));
    
    [pxx_high, ~] = pwelch(sig_high(1, :) - mean(sig_high(1, :)), fs, [], [], fs);
    plot(f, 10*log10(pxx_high), 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Seizure (K=%d)', real_k_high));
    
    xlim([0, 40]);
    title('功率谱密度对比', 'FontSize', 14, 'FontWeight', 'bold');
    xlabel('Frequency (Hz)');
    ylabel('Power (dB)');
    legend('FontSize', 12);
    grid on;
    set(gca, 'FontSize', 12);
    
    % 标注
    text(5, max(10*log10(pxx_high)), '\leftarrow 棘慢波主频', 'Color', 'r', 'FontSize', 12);
    
end