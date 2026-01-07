function NMM()
    % 配置
    output_file = 'nmm_data.mat';
    
    % 扫描耦合强度K的范围
    K_values = 0:2:100; 
    
    % 固定参数
    N_channels = 23;
    fs = 256;
    T_duration = 10;
    n_points = fs * T_duration;
    
    num_k = length(K_values);
    generated_data = zeros(num_k, N_channels, n_points);
    
    fprintf('NMM信号生成器\n');

    for idx = 1:num_k
        K = K_values(idx);
        fprintf('正在仿真 K = %d (%d/%d)...\n', K, idx, num_k);
        
        eeg_segment = run_jansen_rit_23ch(K, N_channels, n_points, 1/fs);
        
        generated_data(idx, :, :) = eeg_segment;
    end
    
    % 保存数据
    fprintf('正在保存至 %s ...\n', output_file);
    save(output_file, 'generated_data', 'K_values', 'fs');
    fprintf('完成！请运行 Python 脚本进行推理。\n');
end

function eeg = run_jansen_rit_23ch(K_coupling, n_ch, n_pts, dt)
    
    A = 3.25; B = 22;
    a = 100;  b = 50;
    C = 135;
    C1 = C; C2 = 0.8*C; C3 = 0.25*C; C4 = 0.25*C;
    v0 = 6; e0 = 2.5; r = 0.56;
    
    S = @(v) 2 * e0 ./ (1 + exp(r * (v0 - v)));
    
    Y = zeros(6, n_ch);
    
    eeg = zeros(n_ch, n_pts);

    coupling_strength = K_coupling / (n_ch - 1);

    noise_input = 120 + 30 * randn(n_ch, n_pts);
    
    for t = 1:n_pts

        y0 = Y(1, :); y1 = Y(2, :); y2 = Y(3, :);
        y3 = Y(4, :); y4 = Y(5, :); y5 = Y(6, :);
        
        pulse_out = S(y1 - y2);
        total_pulse = sum(pulse_out); 
        input_from_others = (total_pulse - pulse_out) * coupling_strength;
        
        dy0 = y3;
        dy3 = A*a*S(y1 - y2) - 2*a*y3 - a^2*y0;
        dy1 = y4;
        p = noise_input(:, t)';
        dy4 = A*a*(p + C2*S(C1*y0) + input_from_others) - 2*a*y4 - a^2*y1;
        dy2 = y5;
        dy5 = B*b*C4*S(C3*y0) - 2*b*y5 - b^2*y2;
        
        dY = [dy0; dy1; dy2; dy3; dy4; dy5];
        Y = Y + dY * dt;
        
        eeg(:, t) = Y(2, :) - Y(3, :);
    end
end