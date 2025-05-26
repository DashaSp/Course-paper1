function prak()
    % Ввод параметров
    m = 1;
    n = 2;
    tau = input('Введите время задержки: ');
    start = input('Введите число, которое начинает последовательность для спектра: ');
    step = input('Введите шаг: ');
    finish = input('Введите конец последовательности: ');
    
    % Инициализация
    R = eye(n);
    A_all = cell(1, m);
    b_all = cell(1, m);
    tol = 1e-6;
    
    % Ввод матриц A и b для каждого режима
    for i = 1:m
        A_all{i} = input(['Введите матрицу A_', num2str(i), ': ']);
        b_all{i} = input(['Введите матрицу b_', num2str(i), ': ']);
        C = ctrb(A_all{i}, b_all{i}); 

        % Проверка ранга матрицы управляемости
        rank_C = rank(C);

        % Если ранг равен числу состояний, система управляема
        if rank_C == size(A_all{i}, 1)
            disp('Система управляема');
        else
            error('Система не управляема');
        end
    end

    if (start < finish) || (start >= 0) || (finish >= 0) || (step >= 0)
        error('Проверьте введённые значения start и finish и step');
    end
    
    row1 = start:step:finish;
    row2 = (start-1):step:(finish-1);
    M = [row1; row2];
    x_values = row1;
    y_values = zeros(size(x_values));
    ans_values_1 = zeros(size(x_values)); % Для первого метода
    ans_values_2 = zeros(size(x_values)); % Для второго метода
    
    figure(1); % График нормы k
    grid on;
    
    figure(2); % График ans_1 и ans_2
    hold on;
    grid on;
    
    for i = 1:size(M, 2)
        l = M(:, i);
        k = place(A_all{1}, b_all{1}, l);        
        y = norm(k, 'fro');
        x = row1(i);
        y_values(i) = y;
        
        % График нормы k
        figure(1);
        plot(x, y, 'bo', 'MarkerSize', 8, 'LineWidth', 1);
        hold on;
        
        % Метод 1: Ляпунов
        H = A_all{1} - b_all{1}*k;
        
        % Проверка устойчивости H
        if any(real(eig(H)) > -tol)
            error('Матрица H неустойчива');
        end
        
        try
            Q = lyap(H', R);
        catch
            error('Ошибка при решении уравнения Ляпунова');
        end
        
        if any(eig(Q) <= 0)
            error('Матрица Q не положительно определена');
        end
        
        lambda = eig(Q);
        M_val = max(lambda);
        m_val = min(lambda);
        rho = sqrt(M_val/m_val);
        theta = 1/(2*M_val);
        ans_values_1(i) = 2*log(rho)/theta;
        
        % Метод 2: Спектральный
        if ~all(abs(imag(eig(H))) < tol)
            error('Комплекстные собственные значения');
        end
        if length(unique(round(real(eig(H)), 10))) < length(eig(H))
            error('Неразличные собственные значения');
        end
        if any(real(eig(H)) > -tol)
            error('Матрица H неустойчива');
        end
        
        gamma = max(abs(eig(H)));
        [T, D] = eig(H);
        row_sum_T = sum(abs(T), 2);
        row_sum_inv_T = sum(abs(inv(T)), 2);
        mu = max(row_sum_inv_T) * max(row_sum_T);
        ans_values_2(i) = 2*log(mu)/gamma;
        
        % График ans_1 и ans_2
        figure(2);
        plot(x, ans_values_1(i), 'ro', 'MarkerSize', 8, 'LineWidth', 1); 
        plot(x, ans_values_2(i), 'gd', 'MarkerSize', 8, 'LineWidth', 1);
    end
    
    % Оформление графиков
    figure(1);
    plot(x_values, y_values, 'b-', 'LineWidth', 1);
    xlabel('n');
    ylabel('|| k ||');
    title('Зависимость нормы k от спектра');
    hold off;
    
    figure(2);
    % Линии и точки для ans_1 (красные)
    plot(x_values, ans_values_1, 'r-', 'LineWidth', 1);
    plot(x_values, ans_values_1, 'ro', 'MarkerSize', 8, 'LineWidth', 1);
    % Линии и точки для ans_2 (зелёные)
    plot(x_values, ans_values_2, 'g--', 'LineWidth', 1);
    plot(x_values, ans_values_2, 'gd', 'MarkerSize', 8, 'LineWidth', 1);
    % tau (черный)
    x_limits = xlim; % Получаем текущие границы x
    plot([x_limits(1), x_limits(2)], [tau tau], '--p', 'LineWidth', 1, 'Color', 'k');
    
    xlabel('n');
    ylabel('tau');
    title('Сравнение методов');
    legend('Метод 1 (Ляпунов)', 'Метод 2 (Спектральный)');
    hold off;
end