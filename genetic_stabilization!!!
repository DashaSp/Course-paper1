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
        [best_k, best_f, history, k_history] = genetic_algorithm(params);
            
        % ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
        plot_convergence(history, k_history, best_k, tau0);
        
        %% 5. АНАЛИЗ РЕЗУЛЬТАТОВ
        fprintf('\n--- ЭТАП 3: АНАЛИЗ РЕЗУЛЬТАТОВ ---\n');
        analyze_results(best_k, best_f, modes, n, tau0);
        
        if best_f >= 0
            fprintf('\n=========================================================\n');
            fprintf('   ОБЩИЙ РЕГУЛЯТОР НЕ НАЙДЕН. НАЙДЕМ K ДЛЯ КАЖДОГО ОБЪЕКТА ПО ОТДЕЛЬНОСТИ\n');
            fprintf('=========================================================\n');
            
            individual_ks = cell(1, m);
            individual_Fs = zeros(1, m);
            individual_success = false(1, m);

            for i = 1:m
                fprintf('\n--- РЕЖИМ %d (ИНДИВИДУАЛЬНЫЙ ПОИСК) ---\n', i);
                A_i = modes(i).A;
                b_i = modes(i).b;

                [k_i, F_i, success_i] = pole_placement_for_single_mode(A_i, b_i, tau0);

                individual_ks{i} = k_i;
                individual_Fs(i) = F_i;
                individual_success(i) = success_i;

                fprintf('Регулятор: k = [');
                fprintf('%.6f ', k_i);
                fprintf(']\n');
                fprintf('F = %.6f\n', F_i);
                if success_i
                    fprintf(' Режим %d стабилизирован индивидуально!\n', i);
                else
                    fprintf(' Режим %d не удалось стабилизировать даже индивидуально.\n', i);
                end
            end
            fprintf('\n=========================================================\n');
            fprintf('   ИТОГИ ИНДИВИДУАЛЬНОГО ПОИСКА\n');
            fprintf('=========================================================\n');
            for i = 1:m
                fprintf('Режим %d: k = [', i);
                fprintf('%.4f ', individual_ks{i});
                fprintf('], F = %.6f', individual_Fs(i));
                if individual_success(i)
                    fprintf(' [УСПЕХ]\n');
                else
                    fprintf(' [НЕУДАЧА]\n');
                end
            end
        end
    end
    
    %% 6. МОДЕЛИРОВАНИЕ (если найден хотя бы один регулятор)
    if success_quad
        simulate_system(modes, k_quad, tau0, n);
    elseif best_f < 0
        simulate_system(modes, best_k, tau0, n);
    else
        % Если общего регулятора нет, но есть успешные индивидуальные,
        % можно смоделировать для первого успешного (для примера)
        if any(individual_success)
            idx = find(individual_success, 1);
            fprintf('\nМоделирование для режима %d с его индивидуальным регулятором:\n', idx);
            simulate_system(modes(idx), individual_ks{idx}, tau0, n);
        end
    end

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
    fprintf('  4 - Демонстративный пример (2 режима)(k ищет генетическим алгоритмом)\n');
    
    choice = input('Ваш выбор (1-4): ');
    
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

        
    elseif choice == 3
        % Демонстрационный пример (2 режима)
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
        
    elseif choice == 4 
        % Демонстрационный пример (3 режима) (k ищет генетиическим алгоритмом)
        m = 2;
        n = 2;
        tau0 = 2.5;
        
        modes = struct('A', cell(1, m), 'b', cell(1, m)); % создаем структуру (массив матриц A и b)
        
        % Режим 1
        modes(1).A = [0, 1; -2, 3];
        modes(1).b = [0; 1];
        
        % Режим 2
        modes(2).A = [1, 0; 1, -2];
        modes(2).b = [1; 0];
    else 
        error('Ошибка при вводе');
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
    params.n = n;                     % размерность
    params.modes = modes;             % режимы системы
    params.tau0 = tau0;               % заданное время задержки
    
    % Параметры популяции
    params.pop_size = 300;            % размер популяции
    params.generations = 10000;         % количество поколений
    
    % Операторы
    params.crossover_rate = 0.8;     % вероятность скрещивания
    params.mutation_rate = 0.5;      % вероятность мутации
    params.mutation_amplitude = 1.5;  % амплитуда мутации
    
    % Границы поиска
    params.k_bounds = [-50, 50];        % границы для k
    
    % Параметры останова
    params.tol_fitness = 1e-6;        % точность по целевой функции
    params.patience = 1000;             % терпение для ранней остановки
    
    % Элитизм
    params.elite_count = 3;           % количество сохраняемых лучших особей
end
%% ==================== ГЕНЕТИЧЕСКИЙ АЛГОРИТМ ====================

function [best_k, best_f, history, k_history] = genetic_algorithm(params)
% Основная функция генетического алгоритма

    % Инициализация
    population = initialize_population(params.pop_size, params.n, params.k_bounds);
    
    best_f = inf;
    best_k = zeros(1, params.n); % Создает матрицу (вектор-строку) размером 1 ? params.n, заполненную нулями.
    no_improve_count = 0;
    
    % Инициализация истории
    history = zeros(params.generations, 1);
    k_history = zeros(params.generations, params.n);
    
    % Заголовок таблицы
    fprintf('\n%-6s | %-12s | %-40s | %-10s\n', ...
            'Покол.', 'Лучшее F', 'Лучший k', 'Статус');
    fprintf('%s\n', repmat('-', 80, 1));
    
    for gen = 1:params.generations
        % Оценка приспособленности
        fitnesses = evaluate_fitness(population, params, gen);
        
        % Поиск лучшей особи
        [current_best_f, best_idx] = min(fitnesses);
        current_best_k = population(best_idx, :);
        
        % Сохраняем в историю
        history(gen) = current_best_f;
        k_history(gen, :) = current_best_k;
        
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
        if mod(gen, 25) == 0 || gen == 1 || current_best_f < 0 % gen - какое поколение
            k_str = sprintf('%.4f ', current_best_k);
            fprintf('%-6d | %-12.6f | %-40s | %-10s\n', gen, current_best_f, ['[' k_str ']'], status);
            
            if current_best_f < 0
                fprintf('%s\n', repmat('-', 80, 1));
                fprintf('УСПЕХ: Найдено стабилизирующее управление на %d поколении!\n', gen);
                fprintf('%s\n', repmat('-', 80, 1));
            end
        end
        
        % Ранняя остановка
        if no_improve_count >= params.patience
            fprintf('\nРанняя остановка на %d поколении (нет улучшений)\n', gen);
            % Обрезаем историю
            history = history(1:gen);
            k_history = k_history(1:gen, :);
            break;
        end
        
        if current_best_f < -0.01 && all(abs(current_best_k) < 1e3)
            % Можно продолжить для улучшения, но пока остановим
            if gen > 49
                history = history(1:gen);
                k_history = k_history(1:gen, :);
                break;
            end
        end
        
        % Селекция
        parents = selection(population, fitnesses, params.pop_size / 2);
        
        % Создание нового поколения
        new_population = create_new_generation(parents, population, fitnesses, params, gen);
        population = new_population;
    end
    
    % Финальный расчет (если не было ранней остановки)
    if length(history) < params.generations
        fitnesses = evaluate_fitness(population, params, gen);
        [best_f, best_idx] = min(fitnesses);
        best_k = population(best_idx, :);
    end
end

%% ==================== ОПЕРАТОРЫ ГЕНЕТИЧЕСКОГО АЛГОРИТМА ====================

function population = initialize_population(pop_size, n, bounds)
% Инициализация популяции случайными векторами

    population = rand(pop_size, n) * (bounds(2) - bounds(1)) + bounds(1);
end

function fitnesses = evaluate_fitness(population, params, gen)
% Вычисление целевой функции для всей популяции

    pop_size = size(population, 1); % Получаем количество особей
    fitnesses = zeros(pop_size, 1); % Создаем вектор для результатов (Создает вектор-столбец из нулей размером pop_size ? 1)
    
    for i = 1:pop_size
        fitnesses(i) = fitness_function(population(i, :), params, gen);
    end
end

function F = fitness_function(k, params, gen)
%   ЦЕЛЕВАЯ ФУНКЦИЯ
%   F = -tau0 + 2/theta * ln(rho), если F1 < 0
%   F = F1, если F1 >= 0

    % Вычисляем F1
    F1 = compute_F1(k, params.modes);
    % Штраф за большие коэффициенты
    % penalty = 0.01 * sum(k.^2); % k.^2 - поэлементное возведение в квадрат 
    penalty = 0.001 * sum(k.^2);
    
    
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
    
    if ~isfinite(tau_res) || tau_res <= 0 % ~isfinite(tau_res) - проверяет, не является ли tau_res бесконечностью (inf) или не числом (NaN)
                                          % tau_res <= 0 - проверяет, положительное ли значение (время задержки должно быть > 0)
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

    pop_size = size(population, 1); % Узнаем, сколько особей в популяции
    num_parents = floor(num_parents); %  Округляет число вниз до ближайшего целого
    parents = zeros(num_parents, size(population, 2)); % Создает матрицу из нулей размером num_parents ? (число столбцов в population)
    
    for i = 1:num_parents
        % Турнир из 3 случайных особей
        idx = randperm(pop_size, 3); % Выбираем 3 случайных разных особи для турнира
        [~, best_idx] = min(fitnesses(idx));
        parents(i, :) = population(idx(best_idx), :); % Сохраняем победителя турнира в список родителей
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

function child = mutation(child, mutation_rate, bounds, amplitude, gen, max_gen)
% МУТАЦИЯ - адаптивная (уменьшается со временем)

    if nargin < 6
        max_gen = 10000;
    end
    
    % Адаптивная амплитуда: больше в начале, меньше в конце
    adaptive_amplitude = amplitude * (1 - 0.5 * gen/max_gen);
    
    for i = 1:length(child)
        if rand() < mutation_rate
            child(i) = child(i) + (rand() - 0.5) * 2 * adaptive_amplitude;
        end
    end
    
    % Ограничение границ
    child = max(min(child, bounds(2)), bounds(1));
end

function new_population = create_new_generation(parents, old_population, fitnesses, params, gen)
% Создание нового поколения с элитизмом

    pop_size = size(old_population, 1); % Определяем размер текущей популяции
    n_vars = size(old_population, 2); % определяем кол-во генов
    new_population = zeros(pop_size, n_vars);
    
    % Элитизм - сохраняем лучшие особи
    [~, sorted_idx] = sort(fitnesses); %  сортирует значения fitness по возрастанию (от лучших к худшим) и возращаем индекты
    elite_count = min(params.elite_count, pop_size); 
    new_population(1:elite_count, :) = old_population(sorted_idx(1:elite_count), :); % сохраняем лучших особей
    
    % Создание остальных особей
    for i = (elite_count + 1):pop_size
        % Выбор родителей
        parent_idx = randperm(size(parents, 1), 2); % Случайно выбираем двух разных родителей для скрещивания
        p1 = parents(parent_idx(1), :);
        p2 = parents(parent_idx(2), :);
        
        % Скрещивание
        child = crossover(p1, p2, params.crossover_rate);
        
        % Мутация
        child = mutation(child, params.mutation_rate, params.k_bounds, params.mutation_amplitude, gen, params.generations);
        
        new_population(i, :) = child;
    end
end
%% ==================== ВИЗУАЛИЗАЦИЯ ====================

function plot_convergence(history, k_history, best_k, tau0)
% График сходимости генетического алгоритма

    figure('Name', 'Сходимость генетического алгоритма', 'Position', [100, 100, 1200, 400]); % размер окна мб подровнять
    
    generations = 1:length(history);
    
    % График значений F
    subplot(1, 2, 1);
    plot(generations, history, 'b.', 'MarkerSize', 10);
    hold on;
    
    % Рисуем горизонтальную линию F = 0
    plot([1, length(history)], [0, 0], 'r--', 'LineWidth', 1.5);
    xlabel('Поколение', 'FontSize', 12);
    ylabel('Значение целевой функции F', 'FontSize', 12);
    title('Сходимость генетического алгоритма', 'FontSize', 14);
    legend('F_{best}', 'F = 0', 'Location', 'best');
    grid on;
    
    % Эволюция коэффициентов k
    subplot(1, 2, 2);
    colors = lines(length(best_k));
    for i = 1:length(best_k)
        plot(generations, k_history(:, i), 'Color', colors(i, :), 'LineWidth', 2, 'DisplayName', sprintf('k_%d', i));
        hold on;
    end
    
    % Рисуем финальные значения как горизонтальные линии
    x_limits = [1, length(history)];
    for i = 1:length(best_k)
        plot(x_limits, [best_k(i), best_k(i)], '--', 'Color', colors(i, :), ...
             'LineWidth', 1, 'HandleVisibility', 'off');
    end
    
    xlabel('Поколение', 'FontSize', 12);
    ylabel('Значение коэффициентов', 'FontSize', 12);
    title('Эволюция коэффициентов регулятора', 'FontSize', 14);
    legend('Location', 'best');
    grid on;
    
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
%% =============== ПОИСК ИНДИВИДУАЛЬНОГО K ====================

function [k, F, success] = pole_placement_for_single_mode(A, b, tau0)
% Находит регулятор для одного режима методом размещения полюсов,
% минимизируя целевую функцию F = -tau0 + min(tau_lyap, tau_diag)

    n = size(A, 1);               % размерность
    best_k = zeros(1, n);
    best_F = inf;
    success = false;

    % Диапазон желаемых полюсов (от медленных до быстрых)
    alpha_min = 0.1;
    alpha_max = 500;
    n_alphas = 500;                % количество точек
    alphas = linspace(alpha_min, alpha_max, n_alphas);

    for alpha = alphas
        poles = [-alpha, -(alpha + 0.1)];

        try
            % Размещение полюсов
            k = place(A, b, poles);
        catch
            % Если place не сработал (например, численные проблемы), пропускаем
            continue;
        end

        % Вычисляем F для этого k
        F = compute_F_single(k, A, b, tau0);

        if F < best_F
            best_F = F;
            best_k = k;
        end
    end

    % Если нашли хоть что-то, проверяем успех и уточняем локально
    if best_F < inf
        % Локальное уточнение вокруг лучшего k
        best_k = local_refinement_single(best_k, A, b, tau0);
        best_F = compute_F_single(best_k, A, b, tau0);
        
        if best_F < 0
            success = true;
        end
    end

    k = best_k;
    F = best_F;
end

function F = compute_F_single(k, A, b, tau0)
% Вычисляет F = -tau0 + min(tau_lyap, tau_diag) для устойчивого режима,
% иначе возвращает max(real(eig)).
    H = A - b * k(:)';
    eigvals = eig(H);
    max_real = max(real(eigvals));

    % Если система неустойчива, возвращаем max_real
    if max_real >= -1e-8
        F = max_real;
        return;
    end

    % Метод Ляпунова
    tau_lyap = inf;
    try
        Q = lyap(H', eye(size(H)));
        Q = (Q + Q')/2;
        lam = eig(Q);
        m = min(lam);
        M = max(lam);
        if m > 0 && M > 0
            rho = sqrt(M/m);
            theta = 1/(2*M);
            tau_lyap = (2/theta) * log(rho);
        end
    end

    % Метод диагонализации
    tau_diag = inf;
    try
        [V, D] = eig(H);
        V_inv = inv(V);
        rho_diag = norm(V, inf) * norm(V_inv, inf);
        theta_diag = -max(real(diag(D)));
        if theta_diag > 0 && rho_diag >= 1
            tau_diag = (2/theta_diag) * log(rho_diag);
        end
    end

    tau_min = min(tau_lyap, tau_diag);
    if isfinite(tau_min) && tau_min > 0
        F = -tau0 + tau_min;
    else
        F = 1e6;   % большой штраф, если оценка не удалась
    end
end

function k = local_refinement_single(k0, A, b, tau0)
% Локальный поиск вокруг начального приближения k0
    k = k0;
    step = 0.1;
    F_current = compute_F_single(k, A, b, tau0);

    for iter = 1:100
        improved = false;
        for i = 1:length(k)
            for delta = [-step, step]
                k_test = k;
                k_test(i) = k(i) + delta;
                F_test = compute_F_single(k_test, A, b, tau0);
                if F_test < F_current
                    k = k_test;
                    F_current = F_test;
                    improved = true;
                end
            end
        end
        if ~improved
            step = step / 2;
        end
        if step < 1e-4
            break;
        end
    end
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

%% ==================== ПРОВЕРКА УСЛОВИЙ СТАБИЛЬНОСТИ ====================

function check_stability_conditions(k, modes, n, tau0)
% Проверка условий устойчивости для найденного регулятора

    fprintf('\n--- ПРОВЕРКА УСЛОВИЙ УСТОЙЧИВОСТИ ---\n');
    
    [tau_lyap, success_lyap] = tau_bound_lyapunov(k, modes, n);
    [tau_diag, success_diag] = tau_bound_diag(k, modes);
    if success_lyap && success_diag
        tau_min = min(tau_lyap, tau_diag);
        fprintf('Метод Ляпунова: 2/theta * ln(rho) = %.6f\n', tau_lyap);
        fprintf('Диагонализация: 2/theta * ln(rho) = %.6f\n', tau_diag);
        fprintf('Минимальная оценка: %.6f\n', tau_min);
        fprintf('Заданное tau0: %.6f\n', tau0);
    else
        if success_lyap
            fprintf('Метод Ляпунова: 2/theta * ln(rho) = %.6f\n', tau_lyap);
            fprintf('Минимальная оценка: %.6f\n', tau_lyap);
            fprintf('Заданное tau0: %.6f\n', tau0);
        end

        if success_diag
            fprintf('Диагонализация: 2/theta * ln(rho) = %.6f\n', tau_diag);
            fprintf('Минимальная оценка: %.6f\n', tau_diag);
            fprintf('Заданное tau0: %.6f\n', tau0);
        end
    end
end

%% ==================== МОДЕЛИРОВАНИЕ СИСТЕМЫ ============================
function simulate_system(modes, k, tau0, n)
    % Моделирование переключаемой системы с найденным регулятором
    % и построение графика нормы состояния.

    % Начальные условия
    x0 = randn(n, 1) * 2;
    fprintf('\n--- МОДЕЛИРОВАНИЕ ПЕРЕКЛЮЧАЕМОЙ СИСТЕМЫ ---\n');
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

    % Построение графика нормы
    figure('Name', 'Сравнение норм состояния', 'Position', [100, 100, 800, 400]);
    norm1 = sqrt(sum(x1.^2, 2));
    norm2 = sqrt(sum(x2.^2, 2));
    semilogy(t1, norm1, 'b-', 'LineWidth', 1.5); hold on;
    semilogy(t2, norm2, 'g-', 'LineWidth', 1.5);
    xlabel('Время t', 'FontSize', 12);
    ylabel('||x(t)||', 'FontSize', 12);
    title('Сравнение норм состояния', 'FontSize', 14);
    legend('Без управления', 'С управлением', 'Location', 'best');
    grid on;
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
    x_current = x0(:);
    options = odeset('RelTol', 1e-6, 'AbsTol', 1e-8);
    for i = 1:length(t_switch)-1
        t_segment = [t_switch(i), t_switch(i+1)];
        mode_idx = mod(i-1, length(modes)) + 1;
        if isempty(k)
            f = @(t, x) modes(mode_idx).A * x;
        else
            f = @(t, x) (modes(mode_idx).A - modes(mode_idx).b * k(:)') * x;
        end
        [t_seg, x_seg] = ode45(f, t_segment, x_current, options);
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
