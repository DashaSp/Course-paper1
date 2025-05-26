function prak()
    % ���� ����������
    m = 1;
    n = 2;
    tau = input('������� ����� ��������: ');
    start = input('������� �����, ������� �������� ������������������ ��� �������: ');
    step = input('������� ���: ');
    finish = input('������� ����� ������������������: ');
    
    % �������������
    R = eye(n);
    A_all = cell(1, m);
    b_all = cell(1, m);
    tol = 1e-6;
    
    % ���� ������ A � b ��� ������� ������
    for i = 1:m
        A_all{i} = input(['������� ������� A_', num2str(i), ': ']);
        b_all{i} = input(['������� ������� b_', num2str(i), ': ']);
        C = ctrb(A_all{i}, b_all{i}); 

        % �������� ����� ������� �������������
        rank_C = rank(C);

        % ���� ���� ����� ����� ���������, ������� ����������
        if rank_C == size(A_all{i}, 1)
            disp('������� ����������');
        else
            error('������� �� ����������');
        end
    end

    if (start < finish) || (start >= 0) || (finish >= 0) || (step >= 0)
        error('��������� �������� �������� start � finish � step');
    end
    
    row1 = start:step:finish;
    row2 = (start-1):step:(finish-1);
    M = [row1; row2];
    x_values = row1;
    y_values = zeros(size(x_values));
    ans_values_1 = zeros(size(x_values)); % ��� ������� ������
    ans_values_2 = zeros(size(x_values)); % ��� ������� ������
    
    figure(1); % ������ ����� k
    grid on;
    
    figure(2); % ������ ans_1 � ans_2
    hold on;
    grid on;
    
    for i = 1:size(M, 2)
        l = M(:, i);
        k = place(A_all{1}, b_all{1}, l);        
        y = norm(k, 'fro');
        x = row1(i);
        y_values(i) = y;
        
        % ������ ����� k
        figure(1);
        plot(x, y, 'bo', 'MarkerSize', 8, 'LineWidth', 1);
        hold on;
        
        % ����� 1: �������
        H = A_all{1} - b_all{1}*k;
        
        % �������� ������������ H
        if any(real(eig(H)) > -tol)
            error('������� H �����������');
        end
        
        try
            Q = lyap(H', R);
        catch
            error('������ ��� ������� ��������� ��������');
        end
        
        if any(eig(Q) <= 0)
            error('������� Q �� ������������ ����������');
        end
        
        lambda = eig(Q);
        M_val = max(lambda);
        m_val = min(lambda);
        rho = sqrt(M_val/m_val);
        theta = 1/(2*M_val);
        ans_values_1(i) = 2*log(rho)/theta;
        
        % ����� 2: ������������
        if ~all(abs(imag(eig(H))) < tol)
            error('������������ ����������� ��������');
        end
        if length(unique(round(real(eig(H)), 10))) < length(eig(H))
            error('����������� ����������� ��������');
        end
        if any(real(eig(H)) > -tol)
            error('������� H �����������');
        end
        
        gamma = max(abs(eig(H)));
        [T, D] = eig(H);
        row_sum_T = sum(abs(T), 2);
        row_sum_inv_T = sum(abs(inv(T)), 2);
        mu = max(row_sum_inv_T) * max(row_sum_T);
        ans_values_2(i) = 2*log(mu)/gamma;
        
        % ������ ans_1 � ans_2
        figure(2);
        plot(x, ans_values_1(i), 'ro', 'MarkerSize', 8, 'LineWidth', 1); 
        plot(x, ans_values_2(i), 'gd', 'MarkerSize', 8, 'LineWidth', 1);
    end
    
    % ���������� ��������
    figure(1);
    plot(x_values, y_values, 'b-', 'LineWidth', 1);
    xlabel('n');
    ylabel('|| k ||');
    title('����������� ����� k �� �������');
    hold off;
    
    figure(2);
    % ����� � ����� ��� ans_1 (�������)
    plot(x_values, ans_values_1, 'r-', 'LineWidth', 1);
    plot(x_values, ans_values_1, 'ro', 'MarkerSize', 8, 'LineWidth', 1);
    % ����� � ����� ��� ans_2 (������)
    plot(x_values, ans_values_2, 'g--', 'LineWidth', 1);
    plot(x_values, ans_values_2, 'gd', 'MarkerSize', 8, 'LineWidth', 1);
    % tau (������)
    x_limits = xlim; % �������� ������� ������� x
    plot([x_limits(1), x_limits(2)], [tau tau], '--p', 'LineWidth', 1, 'Color', 'k');
    
    xlabel('n');
    ylabel('tau');
    title('��������� �������');
    legend('����� 1 (�������)', '����� 2 (������������)');
    hold off;
end