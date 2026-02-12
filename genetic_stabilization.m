function genetic_stabilization()
%% ГЕНЕТИЧЕСКИЙ АЛГОРИТМ СТАБИЛИЗАЦИИ ПЕРЕКЛЮЧАЕМОЙ СИСТЕМЫ
%   Дипломная работа: Спиркина Дарья Олеговна, 414 группа
%   Науч. рук.: Фурсов Андрей Серафимович
%   Кафедра Нелинейных Динамических Систем и Процессов Управления, ВМК МГУ

    clear; clc; close all;
    
    fprintf('=========================================================\n');
    fprintf('  СТАБИЛИЗАЦИЯ ПЕРЕКЛЮЧАЕМОЙ ЛИНЕЙНОЙ СИСТЕМЫ\n');
    fprintf('        С ЗАДАННЫМ ВРЕМЕНЕМ ЗАДЕРЖКИ\n');
    fprintf('=========================================================\n\n');
    
    %% 1. ВВОД ПАРАМЕТРОВ СИСТЕМЫ
    [modes, n, m, tau0] = input_system_parameters();
    
    %% 2. ПРОВЕРКА УПРАВЛЯЕМОСТИ
    check_controllability(modes, n);
    
    %% 3. ПОПЫТКА КВАДРАТИЧНОЙ СТАБИЛИЗАЦИИ (LMI подход)
    fprintf('\n--- ЭТАП 1: ПРОВЕРКА КВАДРАТИЧНОЙ СТАБИЛИЗАЦИИ ---\n');
    [k_quad, success_quad] = quadratic_stabilization(modes, n);
    
    if success_quad
        fprintf('\n✅ КВАДРАТИЧНАЯ СТАБИЛИЗАЦИЯ УСПЕШНА!\n');
        fprintf('   Найден регулятор: k = %s\n', mat2str(k_quad', 4));
        fprintf('   Система будет устойчива при ЛЮБЫХ переключениях\n');
        
        % Проверка времени задержки
        check_stability_conditions(k_quad, modes, n, tau0);
        
        best_k = k_quad;
    else
        fprintf('\n❌ Квадратичная стабилизация НЕВОЗМОЖНА\n');
        fprintf('   Переход к генетическому алгоритму...\n\n');
        
        %% 4. ГЕНЕТИЧЕСКИЙ АЛГОРИТМ
        fprintf('--- ЭТАП 2: ГЕНЕТИЧЕСКИЙ АЛГОРИТМ ---\n');
        
        % Параметры генетического алгоритма
        params = get_default_params(n, modes, tau0);
        
        % Запуск генетического алгоритма
        [best_k, best_f, history, k_history] = genetic_algorithm(params);
        
        %% 5. АНАЛИЗ РЕЗУЛЬТАТОВ
        fprintf('\n--- ЭТАП 3: АНАЛИЗ РЕЗУЛЬТАТОВ ---\n');
        analyze_results(best_k, best_f, modes, n, tau0);
        
        %% 6. ВИЗУАЛИЗАЦИЯ
        visualize_results(history, k_history, best_k, tau0, params);
    end
    
    %% 7. МОДЕЛИРОВАНИЕ
    fprintf('\n--- ЭТАП 4: МОДЕЛИРОВАНИЕ СИСТЕМЫ ---\n');
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
    fprintf('  3 - Пример из диплома (2 режима)\n');
    
    choice = input('Ваш выбор (1-3): ');
    
    if choice == 1
        % Ручной ввод
        m = input('Введите количество режимов m: ');
        tau0 = input('Введите заданное время задержки τ₀: ');
        n = input('Введите размерность системы n: ');
        
        modes = struct('A', cell(1, m), 'b', cell(1, m));
        
        for i = 1:m
            fprintf('\n--- Режим %d ---\n', i);
            A_str = input(['Введите матрицу A_', num2str(i), ' = '], 's');
            b_str = input(['Введите вектор b_', num2str(i), ' = '], 's');
            
            try
                modes(i).A = eval(A_str);
                modes(i).b = eval(b_str);
                
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
        
        modes = struct('A', cell(1, m), 'b', cell(1, m));
        
        % Режим 1
        modes(1).A = [-2, 1; 0, -3];
        modes(1).b = [0; 1];
        
        % Режим 2
        modes(2).A = [-1, 1; 0, -2];
        modes(2).b = [0; 1];
        
        % Режим 3
        modes(3).A = [-3, 0; 1, -1];
        modes(3).b = [1; 0];
        
        fprintf('\nЗагружен демонстрационный пример:\n');
        fprintf('  m = %d, n = %d, τ₀ = %.2f\n', m, n, tau0);
    else
        % Пример из диплома (2 режима)
        m = 2;
        n = 2;
        tau0 = 0.5;
        
        modes = struct('A', cell(1, m), 'b', cell(1, m));
        
        % Режим 1
        modes(1).A = [-4, 0; 0, -2];
        modes(1).b = [1; 8];
        
        % Режим 2
        modes(2).A = [-2, 1; 0, -1];
        modes(2).b = [0; 1];
        
        fprintf('\nЗагружен пример из диплома:\n');
        fprintf('  m = %d, n = %d, τ₀ = %.2f\n', m, n, tau0);
    end
    
    % Вывод информации о системе
    fprintf('\n--- ПАРАМЕТРЫ СИСТЕМЫ ---\n');
    fprintf('Количество режимов: m = %d\n', m);
    fprintf('Размерность: n = %d\n', n);
    fprintf('Заданное время задержки: τ₀ = %.3f\n', tau0);
    
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
        C = ctrb(modes(i).A, modes(i).b);
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
    
    % Проверка наличия YALMIP
    try
        % Создание переменных
        Y = sdpvar(n, n, 'symmetric');
        z = sdpvar(1, n);
        
        % Формирование ограничений
        constraints = [Y >= 1e-6*eye(n)];  % Y > 0
        
        for i = 1:length(modes)
            A = modes(i).A;
            b = modes(i).b;
            
            % Y*A_i' + A_i*Y + z'*b_i' + b_i*z < 0
            LMI_i = Y*A' + A*Y + z'*b' + b*z;
            constraints = [constraints, LMI_i <= -1e-6*eye(n)];
        end
        
        % Настройки решателя
        options = sdpsettings('solver', 'sedumi', 'verbose', 0, ...
                             'cachesolvers', 1);
        
        % Решение
        diagnostics = optimize(constraints, [], options);
        
        if diagnostics.problem == 0
            Y_val = value(Y);
            z_val = value(z);
            
            % Проверка положительной определенности
            if min(eig(Y_val)) > 0
                k = -z_val / Y_val;
                success = true;
            end
        end
    catch ME
        fprintf('  YALMIP не установлен или ошибка: %s\n', ME.message);
        fprintf('  Пропускаем проверку квадратичной стабилизации\n');
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
    params.k_bounds = [-5, 5];        % границы для k
    
    % Параметры останова
    params.tol_fitness = 1e-4;        % точность по целевой функции
    params.patience = 50;             % терпение для ранней остановки
    
    % Элитизм
    params.elite_count = 2;           % количество сохраняемых лучших особей
end

%% ==================== ГЕНЕТИЧЕСКИЙ АЛГОРИТМ ====================

function [best_k, best_f, history, k_history] = genetic_algorithm(params)
% Основная функция генетического алгоритма

    % Инициализация
    population = initialize_population(params.pop_size, params.n, params.k_bounds);
    history = zeros(params.generations, 1);
    k_history = zeros(params.generations, params.n);
    
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
        
        history(gen) = current_best_f;
        k_history(gen, :) = current_best_k;
        
        % Обновление глобального лучшего решения
        if current_best_f < best_f
            best_f = current_best_f;
            best_k = current_best_k;
            no_improve_count = 0;
            
            status = '✓ УЛУЧШЕНИЕ';
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
                fprintf('✅ УСПЕХ: Найдено стабилизирующее управление на %d поколении!\n', gen);
                fprintf('%s\n', repmat('-', 80, 1));
            end
        end
        
        % Ранняя остановка
        if no_improve_count >= params.patience
            fprintf('\nРанняя остановка на %d поколении (нет улучшений)\n', gen);
            history = history(1:gen);
            k_history = k_history(1:gen, :);
            break;
        end
        
        if current_best_f < 0 && all(abs(current_best_k) < 1e3)
            % Можно продолжить для улучшения, но пока остановим
            if gen > 50
                history = history(1:gen);
                k_history = k_history(1:gen, :);
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
%   F = -τ₀ + 2/θ * ln(ρ), если F1 < 0
%   F = F1, если F1 ≥ 0

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
% Вычисление F1 = max_i(max Re(λ(A_i - b_i*k)))

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
%   τ_bound = 2/θ * ln(ρ)

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
        
        % Собственные значения
        lam = eig(Q);
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
    
    if isempty(rho_list) || isempty(theta_list)
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
        [V, D] = eig(H);
        w = diag(D);
        
        % Проверка на различные собственные значения
        diff_mat = abs(w - w');
        diff_mat(eye(length(w)) == 1) = inf;
        
        if min(diff_mat(:)) < tol
            return;
        end
        
        % Максимальная действительная часть
        max_real = max(real(w));
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
    
    % θ = -max_i(θ_i)
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
        fprintf('✅ ЦЕЛЕВАЯ ФУНКЦИЯ F < 0 - УСПЕХ!\n');
    else
        fprintf('❌ ЦЕЛЕВАЯ ФУНКЦИЯ F >= 0 - ТРЕБУЕТСЯ ДОНАСТРОЙКА\n');
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
        
        fprintf('Режим %d: %s (max Re(λ) = %.6f)\n', i, status, max_real);
        fprintf('  Собственные значения:\n');
        for j = 1:length(eigvals)
            if abs(imag(eigvals(j))) < 1e-8
                fprintf('    λ%d = %.6f\n', j, real(eigvals(j)));
            else
                fprintf('    λ%d = %.6f %+.6fi\n', j, real(eigvals(j)), imag(eigvals(j)));
            end
        end
    end
    
    % Оценка времени задержки
    fprintf('\n--- ОЦЕНКА ВРЕМЕНИ ЗАДЕРЖКИ ---\n');
    [tau_lyap, success_lyap] = tau_bound_lyapunov(k, modes, n);
    [tau_diag, success_diag] = tau_bound_diag(k, modes);
    
    fprintf('Метод функции Ляпунова:  ');
    if success_lyap
        fprintf('2/θ * ln(ρ) = %.6f\n', tau_lyap);
    else
        fprintf('ОЦЕНКА НЕВОЗМОЖНА\n');
    end
    
    fprintf('Метод диагонализации:    ');
    if success_diag
        fprintf('2/θ * ln(ρ) = %.6f\n', tau_diag);
    else
        fprintf('ОЦЕНКА НЕВОЗМОЖНА\n');
    end
    
    if success_lyap && success_diag
        tau_min = min(tau_lyap, tau_diag);
        fprintf('МИНИМАЛЬНАЯ ОЦЕНКА:       τ* = %.6f\n', tau_min);
        fprintf('ЗАДАННОЕ ВРЕМЯ:           τ₀ = %.6f\n', tau0);
        
        if tau_min < tau0
            fprintf('✅ УСЛОВИЕ ВЫПОЛНЕНО: τ* < τ₀\n');
            fprintf('   Система стабилизируется с заданным временем задержки\n');
        else
            fprintf('❌ УСЛОВИЕ НЕ ВЫПОЛНЕНО: τ* >= τ₀\n');
            fprintf('   Требуется уменьшить τ₀ до %.6f\n', tau_min * 0.99);
        end
    elseif success_lyap
        fprintf('\n⚠️  Доступна только оценка методом Ляпунова\n');
    elseif success_diag
        fprintf('\n⚠️  Доступна только оценка методом диагонализации\n');
    else
        fprintf('\n❌ НЕ УДАЛОСЬ ПОЛУЧИТЬ ОЦЕНКУ ВРЕМЕНИ ЗАДЕРЖКИ\n');
    end
end

%% ==================== ВИЗУАЛИЗАЦИЯ ====================

function visualize_results(history, k_history, best_k, tau0, params)
% Построение графиков результатов

    figure('Name', 'Результаты генетического алгоритма', ...
           'Position', [100, 100, 1400, 600], ...
           'Color', 'white');
    
    % 1. График сходимости
    subplot(1, 3, 1);
    generations = 1:length(history);
    plot(generations, history, 'b-', 'LineWidth', 2);
    hold on;
    yline(0, 'r--', 'LineWidth', 1.5);
    
    % Отметка успеха
    success_gen = find(history < 0, 1);
    if ~isempty(success_gen)
        scatter(success_gen, history(success_gen), 100, 'g', 'filled', ...
                'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        text(success_gen, history(success_gen), ...
             sprintf('  Успех на %d поколении', success_gen), ...
             'VerticalAlignment', 'bottom', 'FontSize', 10, 'FontWeight', 'bold');
    end
    
    xlabel('Поколение', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Значение целевой функции F', 'FontSize', 12, 'FontWeight', 'bold');
    title('Сходимость генетического алгоритма', 'FontSize', 14, 'FontWeight', 'bold');
    legend('F_{best}', 'F = 0', 'Location', 'best');
    grid on;
    xlim([1, length(history)]);
    
    % 2. Эволюция коэффициентов
    subplot(1, 3, 2);
    colors = lines(params.n);
    for i = 1:params.n
        plot(generations, k_history(:, i), 'Color', colors(i, :), ...
             'LineWidth', 2, 'DisplayName', sprintf('k_%d', i));
        hold on;
    end
    
    % Финальные значения
    for i = 1:params.n
        yline(best_k(i), '--', 'Color', colors(i, :), ...
              'LineWidth', 1, 'HandleVisibility', 'off');
    end
    
    % Границы поиска
    yline(params.k_bounds(1), 'k:', 'LineWidth', 1, 'DisplayName', 'Границы');
    yline(params.k_bounds(2), 'k:', 'LineWidth', 1, 'HandleVisibility', 'off');
    
    xlabel('Поколение', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Значение коэффициентов', 'FontSize', 12, 'FontWeight', 'bold');
    title('Эволюция коэффициентов регулятора', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    xlim([1, length(history)]);
    
    % 3. Финальные значения
    subplot(1, 3, 3);
    axis off;
    
    text_x = 0.1;
    text_y = 0.95;
    line_height = 0.08;
    
    text(text_x, text_y, 'РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ', ...
         'FontSize', 14, 'FontWeight', 'bold', 'Color', 'blue');
    text_y = text_y - line_height * 1.5;
    
    text(text_x, text_y, sprintf('Заданное τ₀ = %.4f', tau0), ...
         'FontSize', 12, 'FontWeight', 'bold');
    text_y = text_y - line_height;
    
    text(text_x, text_y, 'Найденный регулятор:', ...
         'FontSize', 12, 'FontWeight', 'bold');
    text_y = text_y - line_height * 0.8;
    
    for i = 1:params.n
        text(text_x + 0.05, text_y, sprintf('k_%d = %.6f', i, best_k(i)), ...
             'FontSize', 11, 'FontName', 'Courier New');
        text_y = text_y - line_height * 0.7;
    end
    
    text_y = text_y - line_height * 0.5;
    
    [tau_lyap, ~] = tau_bound_lyapunov(best_k, params.modes, params.n);
    [tau_diag, ~] = tau_bound_diag(best_k, params.modes);
    
    if isfinite(tau_lyap)
        text(text_x, text_y, sprintf('Ляпунов: τ* = %.6f', tau_lyap), ...
             'FontSize', 11);
        text_y = text_y - line_height * 0.7;
    end
    
    if isfinite(tau_diag)
        text(text_x, text_y, sprintf('Диагонал.: τ* = %.6f', tau_diag), ...
             'FontSize', 11);
        text_y = text_y - line_height * 0.7;
    end
    
    tau_min = min(tau_lyap, tau_diag);
    if isfinite(tau_min)
        text_y = text_y - line_height * 0.3;
        if tau_min < tau0
            status = '✅ УСПЕХ: τ* < τ₀';
            color = 'green';
        else
            status = '❌ НЕУДАЧА: τ* ≥ τ₀';
            color = 'red';
        end
        text(text_x, text_y, status, 'FontSize', 13, ...
             'FontWeight', 'bold', 'Color', color);
    end
    
    sgtitle(sprintf('ГЕНЕТИЧЕСКИЙ АЛГОРИТМ СТАБИЛИЗАЦИИ (τ₀ = %.2f)', tau0), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    % Сохранение графика
    saveas(gcf, 'genetic_algorithm_results.png');
    fprintf('\nГрафики сохранены в файл genetic_algorithm_results.png\n');
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
    
    % Моделирование без управления
    fprintf('Моделирование без управления...\n');
    [t1, x1] = simulate_switched_system(modes, [], t_span, x0, t_switch);
    
    % Моделирование с управлением
    fprintf('Моделирование с управлением (k = [');
    fprintf('%.4f ', k);
    fprintf('])...\n');
    [t2, x2] = simulate_switched_system(modes, k, t_span, x0, t_switch);
    
    % Визуализация
    plot_simulation_results(t1, x1, t2, x2, t_switch, k, tau0, n);
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
% Моделирование переключаемой системы

    t_out = [];
    x_out = [];
    x_current = x0;
    
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
        
        % Интегрирование
        [t_seg, x_seg] = ode45(f, t_segment, x_current);
        
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
    
    [t_seg, x_seg] = ode45(f, t_segment, x_current);
    t_out = [t_out; t_seg];
    x_out = [x_out; x_seg];
end

function plot_simulation_results(t1, x1, t2, x2, t_switch, k, tau0, n)
% Визуализация результатов моделирования

    figure('Name', 'Моделирование переключаемой системы', ...
           'Position', [150, 150, 1400, 700], ...
           'Color', 'white');
    
    % 1. Фазовый портрет без управления
    subplot(2, 3, 1);
    plot(x1(:, 1), x1(:, 2), 'b-', 'LineWidth', 1.5);
    xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
    title('Фазовый портрет (без управления)', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    axis equal;
    
    % 2. Фазовый портрет с управлением
    subplot(2, 3, 2);
    plot(x2(:, 1), x2(:, 2), 'g-', 'LineWidth', 1.5);
    xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
    title('Фазовый портрет (с управлением)', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    axis equal;
    
    % 3. Сравнение фазовых траекторий
    subplot(2, 3, 3);
    plot(x1(:, 1), x1(:, 2), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Без управления');
    hold on;
    plot(x2(:, 1), x2(:, 2), 'g-', 'LineWidth', 1.5, 'DisplayName', 'С управлением');
    xlabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
    title('Сравнение фазовых траекторий', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    axis equal;
    
    % 4. Переходные процессы x1(t)
    subplot(2, 3, 4);
    plot(t1, x1(:, 1), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Без управления');
    hold on;
    plot(t2, x2(:, 1), 'g-', 'LineWidth', 1.5, 'DisplayName', 'С управлением');
    
    % Отметки переключений
    for i = 1:length(t_switch)
        xline(t_switch(i), 'r--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    end
    
    xlabel('Время t', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('x_1', 'FontSize', 12, 'FontWeight', 'bold');
    title('Переходный процесс x_1(t)', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    xlim([t1(1), t1(end)]);
    
    % 5. Переходные процессы x2(t)
    subplot(2, 3, 5);
    plot(t1, x1(:, 2), 'b-', 'LineWidth', 1.5, 'DisplayName', 'Без управления');
    hold on;
    plot(t2, x2(:, 2), 'g-', 'LineWidth', 1.5, 'DisplayName', 'С управлением');
    
    for i = 1:length(t_switch)
        xline(t_switch(i), 'r--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    end
    
    xlabel('Время t', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('x_2', 'FontSize', 12, 'FontWeight', 'bold');
    title('Переходный процесс x_2(t)', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    xlim([t1(1), t1(end)]);
    
    % 6. Норма состояния
    subplot(2, 3, 6);
    norm1 = sqrt(sum(x1.^2, 2));
    norm2 = sqrt(sum(x2.^2, 2));
    
    semilogy(t1, norm1, 'b-', 'LineWidth', 1.5, 'DisplayName', 'Без управления');
    hold on;
    semilogy(t2, norm2, 'g-', 'LineWidth', 1.5, 'DisplayName', 'С управлением');
    
    for i = 1:length(t_switch)
        xline(t_switch(i), 'r--', 'LineWidth', 0.5, 'HandleVisibility', 'off');
    end
    
    xlabel('Время t', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('||x(t)||', 'FontSize', 12, 'FontWeight', 'bold');
    title('Норма состояния', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    xlim([t1(1), t1(end)]);
    
    sgtitle(sprintf('МОДЕЛИРОВАНИЕ ПЕРЕКЛЮЧАЕМОЙ СИСТЕМЫ (τ₀ = %.2f, k = [%.4f, %.4f])', ...
            tau0, k(1), k(2)), 'FontSize', 16, 'FontWeight', 'bold');
    
    % Сохранение графика
    saveas(gcf, 'simulation_results.png');
    fprintf('Графики моделирования сохранены в файл simulation_results.png\n');
end

%% ==================== ПРОВЕРКА УСЛОВИЙ СТАБИЛЬНОСТИ ====================

function check_stability_conditions(k, modes, n, tau0)
% Проверка условий устойчивости для найденного регулятора

    fprintf('\n--- ПРОВЕРКА УСЛОВИЙ УСТОЙЧИВОСТИ ---\n');
    
    [tau_lyap, success_lyap] = tau_bound_lyapunov(k, modes, n);
    [tau_diag, success_diag] = tau_bound_diag(k, modes);
    
    if success_lyap
        fprintf('Метод Ляпунова: 2/θ * ln(ρ) = %.6f\n', tau_lyap);
    end
    
    if success_diag
        fprintf('Диагонализация: 2/θ * ln(ρ) = %.6f\n', tau_diag);
    end
    
    if success_lyap && success_diag
        tau_min = min(tau_lyap, tau_diag);
        fprintf('Минимальная оценка: %.6f\n', tau_min);
        fprintf('Заданное τ₀: %.6f\n', tau0);
        
        if tau_min < tau0
            fprintf('✅ УСЛОВИЕ ВЫПОЛНЕНО!\n');
        else
            fprintf('❌ УСЛОВИЕ НЕ ВЫПОЛНЕНО\n');
        end
    end
end
