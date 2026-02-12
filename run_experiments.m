function run_experiments()
%% ЗАПУСК ЭКСПЕРИМЕНТОВ ДЛЯ ДИПЛОМА
%   Сравнение эффективности алгоритма при различных τ₀

    fprintf('=========================================================\n');
    fprintf('           ЗАПУСК СЕРИИ ЭКСПЕРИМЕНТОВ\n');
    fprintf('=========================================================\n\n');
    
    % Параметры экспериментов
    tau_values = [0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0];
    n_experiments = 5;  % количество запусков для каждого τ₀
    
    % Тестовая система
    modes = struct('A', cell(1, 3), 'b', cell(1, 3));
    modes(1).A = [-2, 1; 0, -3];
    modes(1).b = [0; 1];
    modes(2).A = [-1, 1; 0, -2];
    modes(2).b = [0; 1];
    modes(3).A = [-3, 0; 1, -1];
    modes(3).b = [1; 0];
    n = 2;
    
    % Таблица результатов
    results = [];
    
    for t = 1:length(tau_values)
        tau0 = tau_values(t);
        fprintf('\n--- ЭКСПЕРИМЕНТЫ ДЛЯ τ₀ = %.2f ---\n', tau0);
        
        success_count = 0;
        tau_estimates = [];
        
        for exp = 1:n_experiments
            fprintf('  Запуск %d/%d... ', exp, n_experiments);
            
            % Параметры генетического алгоритма
            params = struct();
            params.n = n;
            params.modes = modes;
            params.tau0 = tau0;
            params.pop_size = 80;
            params.generations = 200;
            params.crossover_rate = 0.85;
            params.mutation_rate = 0.15;
            params.mutation_amplitude = 0.5;
            params.k_bounds = [-5, 5];
            params.elite_count = 2;
            params.tol_fitness = 1e-4;
            params.patience = 30;
            
            % Запуск алгоритма
            [best_k, best_f, ~, ~] = genetic_algorithm(params);
            
            if best_f < 0
                success_count = success_count + 1;
                
                % Оценка времени задержки
                [tau_lyap, ~] = tau_bound_lyapunov(best_k, modes, n);
                [tau_diag, ~] = tau_bound_diag(best_k, modes);
                tau_min = min(tau_lyap, tau_diag);
                tau_estimates = [tau_estimates; tau_min];
                
                fprintf('УСПЕХ (F = %.4f, τ* = %.4f)\n', best_f, tau_min);
            else
                fprintf('НЕУДАЧА (F = %.4f)\n', best_f);
            end
        end
        
        % Сохраняем результаты
        result = struct();
        result.tau0 = tau0;
        result.success_rate = success_count / n_experiments * 100;
        result.success_count = success_count;
        
        if ~isempty(tau_estimates)
            result.tau_mean = mean(tau_estimates);
            result.tau_std = std(tau_estimates);
            result.tau_min = min(tau_estimates);
            result.tau_max = max(tau_estimates);
        else
            result.tau_mean = NaN;
            result.tau_std = NaN;
            result.tau_min = NaN;
            result.tau_max = NaN;
        end
        
        results = [results; result];
        
        fprintf('\n  Результат для τ₀ = %.2f:\n', tau0);
        fprintf('    Успешных запусков: %d/%d (%.1f%%)\n', ...
                success_count, n_experiments, result.success_rate);
        
        if ~isempty(tau_estimates)
            fprintf('    Средняя оценка τ*: %.4f ± %.4f\n', ...
                    result.tau_mean, result.tau_std);
        end
    end
    
    % Визуализация результатов
    plot_experiment_results(results);
end

function plot_experiment_results(results)
% Построение графиков результатов экспериментов

    figure('Name', 'Результаты экспериментов', ...
           'Position', [100, 100, 1200, 500], ...
           'Color', 'white');
    
    % 1. График успешности
    subplot(1, 2, 1);
    tau_values = [results.tau0];
    success_rates = [results.success_rate];
    
    bar(tau_values, success_rates, 'FaceColor', [0.3, 0.6, 0.9]);
    xlabel('Заданное время задержки τ₀', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Успешность, %', 'FontSize', 12, 'FontWeight', 'bold');
    title('Зависимость успешности от τ₀', 'FontSize', 14, 'FontWeight', 'bold');
    grid on;
    ylim([0, 105]);
    
    for i = 1:length(tau_values)
        text(tau_values(i), success_rates(i) + 2, ...
             sprintf('%.0f%%', success_rates(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
    end
    
    % 2. График оценок времени задержки
    subplot(1, 2, 2);
    
    valid_idx = ~isnan([results.tau_mean]);
    tau_values_valid = [results(valid_idx).tau0];
    tau_means = [results(valid_idx).tau_mean];
    tau_stds = [results(valid_idx).tau_std];
    tau_mins = [results(valid_idx).tau_min];
    tau_maxs = [results(valid_idx).tau_max];
    
    errorbar(tau_values_valid, tau_means, tau_stds, 'bo-', ...
             'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', 'b');
    hold on;
    plot(tau_values_valid, tau_mins, 'g--', 'LineWidth', 1.5);
    plot(tau_values_valid, tau_maxs, 'r--', 'LineWidth', 1.5);
    plot([0, max(tau_values_valid)], [0, max(tau_values_valid)], ...
         'k-', 'LineWidth', 1.5);
    
    xlabel('Заданное время задержки τ₀', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('Оценка τ*', 'FontSize', 12, 'FontWeight', 'bold');
    title('Оценки времени задержки', 'FontSize', 14, 'FontWeight', 'bold');
    legend('Среднее ± σ', 'Минимум', 'Максимум', 'τ* = τ₀', ...
           'Location', 'best');
    grid on;
    axis equal;
    xlim([0, max(tau_values_valid) * 1.1]);
    ylim([0, max([tau_maxs, tau_values_valid]) * 1.1]);
    
    sgtitle('РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Сохранение
    saveas(gcf, 'experiment_results.png');
    fprintf('\nГрафики экспериментов сохранены в файл experiment_results.png\n');
end
