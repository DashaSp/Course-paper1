function genetic_stabilization()
    clear; clc; close all;
    
    fprintf('=========================================================\n');
    fprintf('  СТАБИЛИЗАЦИЯ ПЕРЕКЛЮЧАЕМОЙ ЛИНЕЙНОЙ СИСТЕМЫ\n');
    fprintf('        С ЗАДАННЫМ ВРЕМЕНЕМ ЗАДЕРЖКИ\n');
    fprintf('=========================================================\n\n');
    
    %% 1. ВВОД ПАРАМЕТРОВ СИСТЕМЫ
    [modes, n, m, tau0] = input_system_parameters(); % использовать m для визуализации
    
    %% 2. ПРОВЕРКА УПРАВЛЯЕМОСТИ
    check_controllability(modes, n);
    
    %% 3. ПОПЫТКА КВАДРАТИЧНОЙ СТАБИЛИЗАЦИИ (LMI подход)
    fprintf('\n--- ПРОВЕРКА КВАДРАТИЧНОЙ СТАБИЛИЗАЦИИ ---\n');
    [k_quad, success_quad] = quadratic_stabilization(modes, n);
    
    if success_quad
        fprintf('\n--- КВАДРАТИЧНАЯ СТАБИЛИЗАЦИЯ УСПЕШНА! ---\n');
        fprintf('   Найден регулятор: k = %s\n', mat2str(k_quad', 4));
        fprintf('   Система будет устойчива при ЛЮБЫХ переключениях\n');
        
        % Проверка времени задержки
        check_stability_conditions(k_quad, modes, n, tau0);
        
        best_k = k_quad;
    else
        fprintf('\n Квадратичная стабилизация невозможна!\n');
        fprintf(' Переход к генетическому алгоритму...\n\n');
        
        %% 4. ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
        fprintf('--- ЭТАП 2: ГЕНЕТИЧЕСКИЙ АЛГОРИТМ ---\n');
        
        % Параметры генетического алгоритма
        params = get_default_params(n, modes, tau0);
        
        % Запуск генетического алгоритма
        [best_k, best_f] = genetic_algorithm(params);
        
        %% 5. АНАЛИЗ РЕЗУЛЬТАТОВ
        fprintf('\n--- ЭТАП 3: АНАЛИЗ РЕЗУЛЬТАТОВ ---\n');
        analyze_results(best_k, best_f, modes, n, tau0);
    end
    
    %% 6. МОДЕЛИРОВАНИЕ
    fprintf('\n--- МОДЕЛИРОВАНИЕ СИСТЕМЫ ---\n');
    simulate_system(modes, best_k, tau0, n);
    
    fprintf('\n=========================================================\n');
    fprintf('                      РАБОТА ЗАВЕРШЕНА\n');
    fprintf('=========================================================\n');
end

%% ==================== ФУНКЦИЯ ВВОДА ПАРАМЕТРОВ ==================== 

function [modes, n, m, tau0] = input_system_parameters()
% Ввод параметров системы с возможностью выбора режима ввода
    
    fprintf('Выберите способ ввода данных:\n');
    fprintf('  1 - Ручной ввод\n');
    fprintf('  2 - Демонстрационный пример (3 режима)\n');
    fprintf('  3 - Демонстративный пример (2 режима)\n');
    
    choice = input('Ваш выбор (1-3): ');
    
    if choice == 1
        % Ручной ввод
        m = input('Введите количество режимов m: ');
        tau0 = input('Введите заданное время задержки tau0: ');
        n = input('Введите размерность системы n: ');
        
        modes = struct('A', cell(1, m), 'b', cell(1, m)); % создаем структуру (массив матриц A и b)
        
        for i = 1:m
            fprintf('\n--- Режим %d ---\n', i);
            A_str = input(['Введите матрицу A_', num2str(i), ' = '], 's');
            b_str = input(['Введите вектор b_', num2str(i), ' = '], 's');
            
            try
                modes(i).A = eval(A_str); % преобразует строку в матрицу и помещает в нужную структуру
                modes(i).b = eval(b_str); % преобразует строку в матрицу и помещает в нужную структуру
                
                % Проверка размерности
                if size(modes(i).A, 1) ~= n || size(modes(i).A, 2) ~= n 
                    error('Матрица A должна быть размером %dx%d', n, n);
                end
                if length(modes(i).b) ~= n 
                    error('Вектор b должен быть размером %d', n);
                end
            catch ME
                error('Ошибка ввода: %s', ME.message);
            end
        end
        
    elseif choice == 2
        % Демонстрационный пример (3 режима)
        m = 3;
        n = 2;
        tau0 = 0.5;
        
        modes = struct('A', cell(1, m), 'b', cell(1, m)); % создаем структуру (массив матриц A и b)
        
        % Режим 1
        modes(1).A = [-2, 1; 0, -3];
        modes(1).b = [0; 1];
        
        % Режим 2
        modes(2).A = [-1, 1; 0, -2];
        modes(2).b = [0; 1];
        
        % Режим 3
        modes(3).A = [-3, 0; 1, -1];
        modes(3).b = [1; 0];

        
    else
        % Пример из диплома (2 режима)
        m = 2;
        n = 2;
        tau0 = 0.5;
        
        modes = struct('A', cell(1, m), 'b', cell(1, m)); % создаем структуру (массив матриц A и b)
        
        % Режим 1
        modes(1).A = [-4, 0; 0, -2];
        modes(1).b = [1; 8];
        
        % Режим 2
        modes(2).A = [-2, 1; 0, -1];
        modes(2).b = [0; 1];
        
    end
    
    % Вывод информации о системе
    fprintf('\n--- ВЫВОД ИНФОРМАЦИИ О СИСТЕМЕ ---\n');
    fprintf('Количество режимов: m = %d\n', m);
    fprintf('Размерность: n = %d\n', n);
    fprintf('Заданное время задержки: tau0 = %.3f\n', tau0);
    
    for i = 1:m
        fprintf('\nРежим %d:\n', i);
        fprintf('  A_%d = ', i); disp(modes(i).A);
        fprintf('  b_%d = ', i); disp(modes(i).b');
    end
end

%% ==================== ПРОВЕРКА УПРАВЛЯЕМОСТИ ====================

function check_controllability(modes, n)
% Проверка управляемости для каждого режима

    fprintf('\n--- ПРОВЕРКА УПРАВЛЯЕМОСТИ ---\n');
    all_controllable = true;
    
    for i = 1:length(modes)
        C = ctrb(modes(i).A, modes(i).b); % составляем матрицу управляемости 
        rank_C = rank(C);
        
        if rank_C == n
            fprintf('  Режим %d: УПРАВЛЯЕМ (rank = %d)\n', i, rank_C);
        else
            fprintf('  Режим %d: НЕ УПРАВЛЯЕМ (rank = %d/%d)\n', i, rank_C, n);
            all_controllable = false;
        end
    end
    
    if ~all_controllable
        error('Ошибка: Система не полностью управляема');
    end
end

%% ==================== КВАДРАТИЧНАЯ СТАБИЛИЗАЦИЯ (LMI) ====================

function [k, success] = quadratic_stabilization(modes, n)
% Метод квадратичной стабилизации через LMI

    success = false;
    k = [];
    
    Y = sdpvar(n, n, 'symmetric'); % создания симметричной матричной переменной
    z = sdpvar(1, n); %создание матрицы-строки
    
    % Формирование ограничений
    constraints = [Y >= 1e-6*eye(n)];  % Y > 0
    
    for i = 1:length(modes)
        A = modes(i).A;
        b = modes(i).b;
        
        % Y*A_i' + A_i*Y + z'*b_i' + b_i*z < 0
        LMI_i = Y*A' + A*Y + z'*b' + b*z;
        constraints = [constraints, LMI_i <= -1e-6*eye(n)]; % добавляем ограничение(знак меньше)
    end
    
    % Настройки решателя
    options = sdpsettings('solver', 'sedumi', 'verbose', 0, 'cachesolvers', 1); 
                     %Эта команда создает структуру с настройками для решения задачи оптимизации в YALMIP:
                     %'solver', 'sedumi' — Использовать конкретный решатель SeDuMi.
                     %'verbose', 0 — Работать в тихом режиме (не выводить лог решения в консоль).
                     %'cachesolvers', 1 — Кэшировать список доступных солверов для ускорения повторных запусков.
    
    % Решение
    diagnostics = optimize(constraints, [], options); % "[]" означает, что у задачи нет целевой функции
    
    if diagnostics.problem == 0
        Y_val = value(Y); % функция YALMIP, которая извлекает численные значения переменных после оптимизации
        z_val = value(z);
        
        % Проверка положительной определенности
        if min(eig(Y_val)) > 0
            k = -z_val / Y_val; % k = -zY^(-1)
            success = true;
        end
    end
end

%% ==================== ПАРАМЕТРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА ====================

function params = get_default_params(n, modes, tau0)
% Параметры генетического алгоритма по умолчанию

    params = struct();
    params.n = n;                       % размерность
    params.modes = modes;              % режимы системы
    params.tau0 = tau0;               % заданное время задержки
    
    % Параметры популяции
    params.pop_size = 100;            % размер популяции
    params.generations = 300;         % количество поколений
    
    % Операторы
    params.crossover_rate = 0.85;     % вероятность скрещивания
    params.mutation_rate = 0.15;      % вероятность мутации
    params.mutation_amplitude = 0.5;  % амплитуда мутации
    
    % Границы поиска
    params.k_bounds = [-10, 10];        % границы для k
    
    % Параметры останова
    params.tol_fitness = 1e-4;        % точность по целевой функции
    params.patience = 50;             % терпение для ранней остановки
    
    % Элитизм
    params.elite_count = 2;           % количество сохраняемых лучших особей
end

%% ==================== ГЕНЕТИЧЕСКИЙ АЛГОРИТМ ====================

function [best_k, best_f] = genetic_algorithm(params)
% Основная функция генетического алгоритма

    % Инициализация
    population = initialize_population(params.pop_size, params.n, params.k_bounds);
    
    best_f = inf;
    best_k = zeros(1, params.n);
    no_improve_count = 0;
    
    % Заголовок таблицы
    fprintf('\n%-6s | %-12s | %-40s | %-10s\n', ...
            'Покол', 'Лучшее F', 'Лучший k', 'Статус');
    fprintf('%s\n', repmat('-', 80, 1));
    
    for gen = 1:params.generations
        % Оценка приспособленности
        fitnesses = evaluate_fitness(population, params);
        
        % Поиск лучшей особи
        [current_best_f, best_idx] = min(fitnesses);
        current_best_k = population(best_idx, :);
        
        % Обновление глобального лучшего решения
        if current_best_f < best_f
            best_f = current_best_f;
            best_k = current_best_k;
            no_improve_count = 0;
            
            status = 'УЛУЧШЕНИЕ';
        else
            no_improve_count = no_improve_count + 1;
            status = '';
        end
        
        % Вывод прогресса
        if mod(gen, 25) == 0 || gen == 1 || current_best_f < 0
            k_str = sprintf('%.4f ', current_best_k);
            fprintf('%-6d | %-12.6f | %-40s | %-10s\n', ...
                    gen, current_best_f, ['[' k_str ']'], status);
            
            if current_best_f < 0
                fprintf('%s\n', repmat('-', 80, 1));
                fprintf('УСПЕХ: Найдено стабилизирующее управление на %d поколении!\n', gen);
                fprintf('%s\n', repmat('-', 80, 1));
            end
        end
        
        % Ранняя остановка
        if no_improve_count >= params.patience
            fprintf('\nРанняя остановка на %d поколении (нет улучшений)\n', gen);
            break;
        end
        
        if current_best_f < 0 && all(abs(current_best_k) < 1e3)
            % Можно продолжить для улучшения, но пока остановим
            if gen > 50
                break;
            end
        end
        
        % Селекция
        parents = selection(population, fitnesses, params.pop_size / 2);
        
        % Создание нового поколения
        new_population = create_new_generation(parents, population, fitnesses, params);
        population = new_population;
    end
    
    % Финальный расчет
    fitnesses = evaluate_fitness(population, params);
    [best_f, best_idx] = min(fitnesses);
    best_k = population(best_idx, :);
end

%% ==================== ОПЕРАТОРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА ====================

function population = initialize_population(pop_size, n, bounds)
% Инициализация популяции случайными векторами

    population = rand(pop_size, n) * (bounds(2) - bounds(1)) + bounds(1);
end

function fitnesses = evaluate_fitness(population, params)
% Вычисление целевой функции для всей популяции

    pop_size = size(population, 1);
    fitnesses = zeros(pop_size, 1);
    
    for i = 1:pop_size
        fitnesses(i) = fitness_function(population(i, :), params);
    end
end

function F = fitness_function(k, params)
% ЦЕЛЕВАЯ ФУНКЦИЯ
%   F = -tau0 + 2/theta * ln(rho), если F1 < 0
%   F = F1, если F1 >= 0

    % Вычисляем F1
    F1 = compute_F1(k, params.modes);
    
    % Штраф за большие коэффициенты
    penalty = 0.01 * sum(k.^2);
    
    if F1 >= -1e-6  % Есть неустойчивые режимы
        F = F1 + penalty;
        return;
    end
    
    % Оценка времени задержки методом Ляпунова
    [tau_lyap, valid_lyap] = tau_bound_lyapunov(k, params.modes, params.n);
    
    % Оценка времени задержки методом диагонализации
    [tau_diag, valid_diag] = tau_bound_diag(k, params.modes);
    
    % Выбор наилучшей оценки
    tau_res = inf;
    if valid_lyap && valid_diag
        tau_res = min(tau_lyap, tau_diag);
    elseif valid_lyap
        tau_res = tau_lyap;
    elseif valid_diag
        tau_res = tau_diag;
    end
    
    if ~isfinite(tau_res) || tau_res <= 0
        F = 1e6 + penalty;  % Штраф
    else
        F = -params.tau0 + tau_res + penalty;
    end
end

function F1 = compute_F1(k, modes)
% Вычисление F1 = max_i(max Re(lambda(A_i - b_i*k)))

    F1 = -inf;
    for i = 1:length(modes)
        H = modes(i).A - modes(i).b * k(:)';
        eigvals = eig(H);
        max_real = max(real(eigvals));
        F1 = max(F1, max_real);
    end
end

function parents = selection(population, fitnesses, num_parents)
% СЕЛЕКЦИЯ - турнирная

    pop_size = size(population, 1);
    num_parents = floor(num_parents);
    parents = zeros(num_parents, size(population, 2));
    
    for i = 1:num_parents
        % Турнир из 3 случайных особей
        idx = randperm(pop_size, 3);
        [~, best_idx] = min(fitnesses(idx));
        parents(i, :) = population(idx(best_idx), :);
    end
end

function child = crossover(p1, p2, crossover_rate)
% КРОССОВЕР - арифметический

    if rand() < crossover_rate
        alpha = rand();
        child = alpha * p1 + (1 - alpha) * p2;
    else
        child = p1;
    end
end

function child = mutation(child, mutation_rate, bounds, amplitude)
% МУТАЦИЯ - равномерная

    if nargin < 4
        amplitude = 0.5;
    end
    
    for i = 1:length(child)
        if rand() < mutation_rate
            child(i) = child(i) + (rand() - 0.5) * 2 * amplitude;
        end
    end
    
    % Ограничение границ
    child = max(min(child, bounds(2)), bounds(1));
end

function new_population = create_new_generation(parents, old_population, fitnesses, params)
% Создание нового поколения с элитизмом

    pop_size = size(old_population, 1);
    n_vars = size(old_population, 2);
    new_population = zeros(pop_size, n_vars);
    
    % Элитизм - сохраняем лучшие особи
    [~, sorted_idx] = sort(fitnesses);
    elite_count = min(params.elite_count, pop_size);
    new_population(1:elite_count, :) = old_population(sorted_idx(1:elite_count), :);
    
    % Создание остальных особей
    for i = (elite_count + 1):pop_size
        % Выбор родителей
        parent_idx = randperm(size(parents, 1), 2);
        p1 = parents(parent_idx(1), :);
        p2 = parents(parent_idx(2), :);
        
        % Скрещивание
        child = crossover(p1, p2, params.crossover_rate);
        
        % Мутация
        child = mutation(child, params.mutation_rate, ...
                        params.k_bounds, params.mutation_amplitude);
        
        new_population(i, :) = child;
    end
end

%% ==================== МЕТОД ФУНКЦИИ ЛЯПУНОВА ====================

function [tau_bound, success] = tau_bound_lyapunov(k, modes, n)
% ОЦЕНКА ВРЕМЕНИ ЗАДЕРЖКИ МЕТОДОМ ФУНКЦИИ ЛЯПУНОВА
%   tau_bound = 2/theta * ln(rho)

    tau_bound = inf;
    success = false;
    
    rho_list = [];
    theta_list = [];
    
    for i = 1:length(modes)
        H = modes(i).A - modes(i).b * k(:)';
        
        % Проверка устойчивости
        if max(real(eig(H))) >= -1e-6
            return;
        end
        
        % Решение уравнения Ляпунова
        try
            Q = lyap(H', eye(n));
        catch
            return;
        end
        
        % Симметризация
        Q = (Q + Q') / 2;
        
        lam = eig(Q); % Собственные значения
        m_val = min(lam);
        M = max(lam);
        
        if m_val <= 1e-6 || M <= 1e-6
            return;
        end
        
        c = sqrt(M / m_val);
        nu = 1.0 / (2.0 * M);
        
        rho_list = [rho_list; c];
        theta_list = [theta_list; nu];
    end
    
    if isempty(rho_list) || isempty(theta_list) % Если хотя бы один из списков пуст - выходим из функции
        return;
    end
    
    rho = max(rho_list);
    theta = min(theta_list);
    
    if theta > 1e-6 && rho >= 1
        tau_bound = (2.0 / theta) * log(rho);
        success = true;
    end
end

%% ==================== МЕТОД ДИАГОНАЛИЗАЦИИ ====================

function [tau_bound, success] = tau_bound_diag(k, modes, tol)
% ОЦЕНКА ВРЕМЕНИ ЗАДЕРЖКИ МЕТОДОМ ДИАГОНАЛИЗАЦИИ

    if nargin < 3
        tol = 1e-8;
    end
    
    tau_bound = inf;
    success = false;
    
    rho_list = [];
    theta_i_list = [];
    
    for i = 1:length(modes)
        H = modes(i).A - modes(i).b * k(:)';
        
        % Проверка устойчивости
        if max(real(eig(H))) >= -1e-6
            return;
        end
        
        % Собственные значения и вектора
        [V, D] = eig(H); %V - матрица собственных векторов (каждый столбец - собственный вектор)
                         %D - диагональная матрица с собственными значениями
        w = diag(D);
        
        % Проверка на различные собственные значения
        diff_mat = abs(w - w');
        diff_mat(eye(length(w)) == 1) = inf;
        
        if min(diff_mat(:)) < tol
            return;
        end
        
        % находим theta_i
        max_real = max(real(w)); % Максимальная действительная часть
        theta_i_list = [theta_i_list; max_real];
        
        % Вычисление числа обусловленности
        try
            V_inv = inv(V);
        catch
            return;
        end
        
        % Норма бесконечности
        rho_i = norm(V, inf) * norm(V_inv, inf);
        
        if isfinite(rho_i) && rho_i >= 1
            rho_list = [rho_list; rho_i];
        end
    end
    
    if isempty(theta_i_list) || isempty(rho_list)
        return;
    end
    
    % theta = -max_i(theta_i)
    theta = -max(theta_i_list);
    
    if theta <= 1e-6
        return;
    end
    
    rho = max(rho_list);
    tau_bound = (2.0 / theta) * log(rho);
    success = true;
end

%% ==================== АНАЛИЗ РЕЗУЛЬТАТОВ ====================

function analyze_results(k, F, modes, n, tau0)
% Анализ найденного регулятора

    fprintf('\n--- АНАЛИЗ НАЙДЕННОГО РЕГУЛЯТОРА ---\n');
    fprintf('k* = [');
    for i = 1:length(k)
        fprintf('%.6f ', k(i));
    end
    fprintf(']\n');
    fprintf('F(k*) = %.6f\n', F);
    
    if F < 0
        fprintf('ЦЕЛЕВАЯ ФУНКЦИЯ F < 0 - УСПЕХ!\n');
    else
        fprintf('ЦЕЛЕВАЯ ФУНКЦИЯ F >= 0 - ТРЕБУЕТСЯ ДОНАСТРОЙКА\n');
    end
    
    % Проверка устойчивости режимов
    fprintf('\n--- УСТОЙЧИВОСТЬ РЕЖИМОВ ---\n');
    all_stable = true;
    
    for i = 1:length(modes)
        H = modes(i).A - modes(i).b * k(:)';
        eigvals = eig(H);
        max_real = max(real(eigvals));
        
        if max_real < 0
            status = 'УСТОЙЧИВ';
        else
            status = 'НЕУСТОЙЧИВ';
            all_stable = false;
        end
        
        fprintf('Режим %d: %s (max Re(lambda) = %.6f)\n', i, status, max_real);
        fprintf('  Собственные значения:\n');
        for j = 1:length(eigvals)
            if abs(imag(eigvals(j))) < 1e-8
                fprintf('    lambda%d = %.6f\n', j, real(eigvals(j)));
            else
                fprintf('    lambda%d = %.6f %+.6fi\n', j, real(eigvals(j)), imag(eigvals(j)));
            end
        end
    end
    
    % Оценка времени задержки
    fprintf('\n--- ОЦЕНКА ВРЕМЕНИ ЗАДЕРЖКИ ---\n');
    [tau_lyap, success_lyap] = tau_bound_lyapunov(k, modes, n);
    [tau_diag, success_diag] = tau_bound_diag(k, modes);
    
    fprintf('Метод функции Ляпунова:  ');
    if success_lyap
        fprintf('2/theta * ln(rho) = %.6f\n', tau_lyap);
    else
        fprintf('ОЦЕНКА НЕВОЗМОЖНА\n');
    end
    
    fprintf('Метод диагонализации:    ');
    if success_diag
        fprintf('2/theta * ln(rho) = %.6f\n', tau_diag);
    else
        fprintf('ОЦЕНКА НЕВОЗМОЖНА\n');
    end
    
    if success_lyap && success_diag
        tau_min = min(tau_lyap, tau_diag);
        fprintf('МИНИМАЛЬНАЯ ОЦЕНКА:       tau* = %.6f\n', tau_min);
        fprintf('ЗАДАННОЕ ВРЕМЯ:           tau0 = %.6f\n', tau0);
        
        if tau_min < tau0
            fprintf('УСЛОВИЕ ВЫПОЛНЕНО: tau* < tau0\n');
            fprintf('   Система стабилизируется с заданным временем задержки\n');
        else
            fprintf('УСЛОВИЕ НЕ ВЫПОЛНЕНО: tau* >= tau0\n');
            fprintf('   Требуется уменьшить tau0 до %.6f\n', tau_min * 0.99);
        end
    elseif success_lyap
        fprintf('\nДоступна только оценка методом Ляпунова\n');
    elseif success_diag
        fprintf('\nДоступна только оценка методом диагонализации\n');
    else
        fprintf('\nНЕ УДАЛОСЬ ПОЛУЧИТЬ ОЦЕНКУ ВРЕМЕНИ ЗАДЕРЖКИ\n');
    end
end

%% ==================== МОДЕЛИРОВАНИЕ СИСТЕМЫ ====================

function simulate_system(modes, k, tau0, n)
% Моделирование переключаемой системы с найденным регулятором

    fprintf('\n--- МОДЕЛИРОВАНИЕ ПЕРЕКЛЮЧАЕМОЙ СИСТЕМЫ ---\n');
    
    % Начальные условия
    x0 = randn(n, 1) * 2;
    fprintf('Начальное состояние: x0 = [');
    fprintf('%.2f ', x0);
    fprintf(']\n');
    
    % Время моделирования
    t_span = [0, 30];
    
    % Генерация переключающего сигнала
    t_switch = generate_switching_times(t_span, tau0);
    fprintf('Сгенерировано %d переключений\n', length(t_switch)-1);
    
    % Моделирование без управления
    fprintf('Моделирование без управления...\n');
    [t1, x1] = simulate_switched_system(modes, [], t_span, x0, t_switch);
    
    % Моделирование с управлением
    fprintf('Моделирование с управлением (k = [');
    fprintf('%.4f ', k);
    fprintf('])...\n');
    [t2, x2] = simulate_switched_system(modes, k, t_span, x0, t_switch);
    
    % Вывод результатов моделирования
    fprintf('\n--- РЕЗУЛЬТАТЫ МОДЕЛИРОВАНИЯ ---\n');
    
    % Норма состояния в конце
    norm1_end = norm(x1(end, :));
    norm2_end = norm(x2(end, :));
    fprintf('Норма состояния в конце:\n');
    fprintf('  Без управления: ||x|| = %.6f\n', norm1_end);
    fprintf('  С управлением:  ||x|| = %.6f\n', norm2_end);
    
    % Максимальная норма
    norm1_max = max(sqrt(sum(x1.^2, 2)));
    norm2_max = max(sqrt(sum(x2.^2, 2)));
    fprintf('Максимальная норма:\n');
    fprintf('  Без управления: ||x||_max = %.6f\n', norm1_max);
    fprintf('  С управлением:  ||x||_max = %.6f\n', norm2_max);
    
    % Интегральная оценка
    int1 = trapz(t1, sqrt(sum(x1.^2, 2)));
    int2 = trapz(t2, sqrt(sum(x2.^2, 2)));
    fprintf('Интегральная норма (?||x|| dt):\n');
    fprintf('  Без управления: %.6f\n', int1);
    fprintf('  С управлением:  %.6f\n', int2);
    
    % Время достижения порога (например, 1% от начальной нормы)
    threshold = 0.01 * norm(x0);
    t_settle1 = find_settling_time(t1, x1, threshold);
    t_settle2 = find_settling_time(t2, x2, threshold);
    
    fprintf('Время достижения 1%% от начальной нормы:\n');
    if ~isinf(t_settle1)
        fprintf('  Без управления: t = %.2f\n', t_settle1);
    else
        fprintf('  Без управления: не достигнут за время моделирования\n');
    end
    if ~isinf(t_settle2)
        fprintf('  С управлением:  t = %.2f\n', t_settle2);
    else
        fprintf('  С управлением:  не достигнут за время моделирования\n');
    end
    
    % Сравнение
    fprintf('\nСРАВНЕНИЕ ЭФФЕКТИВНОСТИ УПРАВЛЕНИЯ:\n');
    
    if int2 > 0
        fprintf('  Улучшение интегральной нормы: %.2f раз\n', int1/int2);
    else
        fprintf('  Интегральная норма с управлением близка к нулю\n');
    end
    
    if t_settle2 > 0 && ~isinf(t_settle2) && t_settle1 > 0 && ~isinf(t_settle1)
        fprintf('  Ускорение переходного процесса: %.2f раз\n', t_settle1/t_settle2);
    end
    
    % Проверка устойчивости по норме в конце
    if norm2_end < 1e-6
        fprintf('  Система стабилизировалась (норма близка к нулю)\n');
    end
end

function t_settle = find_settling_time(t, x, threshold)
% Находит время, когда норма становится меньше порога
    norm_x = sqrt(sum(x.^2, 2));
    idx = find(norm_x < threshold, 1);
    if isempty(idx)
        t_settle = inf;
    else
        t_settle = t(idx);
    end
end

function t_switch = generate_switching_times(t_span, tau0)
% Генерация моментов переключений

    t_start = t_span(1);
    t_end = t_span(2);
    
    t_switch = t_start;
    t_current = t_start;
    
    while t_current < t_end
        dt = tau0 + rand() * tau0;  % случайное время работы режима
        t_current = t_current + dt;
        if t_current < t_end
            t_switch = [t_switch, t_current];
        end
    end
end

function [t_out, x_out] = simulate_switched_system(modes, k, t_span, x0, t_switch)
% Моделирование переключаемой системы (улучшенная версия)

    t_out = [];
    x_out = [];
    x_current = x0(:);  % гарантируем столбец
    
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);  % повышаем точность
    
    for i = 1:length(t_switch)-1
        t_segment = [t_switch(i), t_switch(i+1)];
        
        % Определение текущего режима
        mode_idx = mod(i-1, length(modes)) + 1;
        
        % Функция правой части
        if isempty(k)
            f = @(t, x) modes(mode_idx).A * x;
        else
            f = @(t, x) (modes(mode_idx).A - modes(mode_idx).b * k(:)') * x;
        end
        
        % Интегрирование с повышенной точностью
        [t_seg, x_seg] = ode45(f, t_segment, x_current, options);
        
        % Сохраняем результаты (кроме последней точки, чтобы избежать дублирования)
        t_out = [t_out; t_seg(1:end-1)];
        x_out = [x_out; x_seg(1:end-1, :)];
        
        x_current = x_seg(end, :)';
    end
    
    % Последний сегмент
    t_segment = [t_switch(end), t_span(2)];
    mode_idx = mod(length(t_switch)-1, length(modes)) + 1;
    
    if isempty(k)
        f = @(t, x) modes(mode_idx).A * x;
    else
        f = @(t, x) (modes(mode_idx).A - modes(mode_idx).b * k(:)') * x;
    end
    
    [t_seg, x_seg] = ode45(f, t_segment, x_current, options);
    t_out = [t_out; t_seg];
    x_out = [x_out; x_seg];
end

%% ==================== ПРОВЕРКА УСЛОВИЙ СТАБИЛЬНОСТИ ====================

function check_stability_conditions(k, modes, n, tau0)
% Проверка условий устойчивости для найденного регулятора

    fprintf('\n--- ПРОВЕРКА УСЛОВИЙ УСТОЙЧИВОСТИ ---\n');
    
    [tau_lyap, success_lyap] = tau_bound_lyapunov(k, modes, n);
    [tau_diag, success_diag] = tau_bound_diag(k, modes);
    
    if success_lyap
        fprintf('Метод Ляпунова: 2/theta * ln(rho) = %.6f\n', tau_lyap);
    end
    
    if success_diag
        fprintf('Диагонализация: 2/theta * ln(rho) = %.6f\n', tau_diag);
    end
    
    if success_lyap && success_diag
        tau_min = min(tau_lyap, tau_diag);
        fprintf('Минимальная оценка: %.6f\n', tau_min);
        fprintf('Заданное tau0: %.6f\n', tau0);
        
        if tau_min < tau0
            fprintf('УСЛОВИЕ ВЫПОЛНЕНО!\n');
        else
            fprintf('УСЛОВИЕ НЕ ВЫПОЛНЕНО\n');
        end
    else
        if success_lyap
            fprintf('Заданное tau0: %.6f\n', tau0);
            fprintf('УСЛОВИЕ ВЫПОЛНЕНО!\n');
        end
    end
end
