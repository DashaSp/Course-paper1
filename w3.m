function w3()
    % ���� ����������
    m = input('������� ���������� �������: ');
    tau = input('������� ����� ��������: ');
    n = input('������� ����������� k: ');
    
    % �������������
    R = eye(n);
    A_all = cell(1, m);
    b_all = cell(1, m);
    k_1 = [];
    k_2 = [];
    theta_1 = Inf;
    theta_2 = Inf;
    tol = 1e-6;
    
    % ���� ������ A � b ��� ������� ������
    for i = 1:m
        A_all{i} = input(['������� ������� A_', num2str(i)]);
        b_all{i} = input(['������� ������� b_', num2str(i)]);
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

    start = -500; 
    finish = 500; 
    step = 0.1;
    
    [grids{1:n}] = ndgrid(start:step:finish);
    k_combinations = reshape(cat(n+1, grids{:}), [], n);
    
    % ������� ������� ��������
    for j = 1:size(k_combinations, 1)
        k = k_combinations(j, :)';
        is_valid = true;
        rho = 0;
        theta = Inf;
        
        % �������� ��� ������� ������
        for i = 1:m
            H = A_all{i} - b_all{i}*k';
            
            % �������� ������������ H
            if any(real(eig(H)) >= 0)
                is_valid = false;
                break;
            end
            
            % ������� ��������� ��������
            try
                Q = lyap(H', R);
            catch
                is_valid = false;
                break;
            end
            
            % �������� ������������� �������������� Q
            if any(eig(Q) <= 0)
                is_valid = false;
                break;
            end
            
            % ���������� ����������
            lambda = eig(Q);
            M = max(lambda);
            m_val = min(lambda);
            rho = max(rho, sqrt(M/m_val));
            theta = min(theta, 1/(2*M));
        end
        
        if is_valid  
            if tau > (2*log(rho)/theta)
                disp(['���������� k: ', mat2str(k')]);
                disp(['rho = ', num2str(rho), ', theta = ', num2str(theta)]);
                
                if (theta_1 == 0) || (2*log(rho)/theta) < theta_1
                    theta_1 = 2*log(rho)/theta;
                    k_1 = k;
                end
            end
        end
    end
    
    if ~isempty(k_1)
        disp(['����������� k_1: ', mat2str(k_1')]);
        disp(['����������� 2*log(rho)/theta: ', num2str(theta_1)]);
    else
        disp('���������� k �� �������');
    end
    
    % 2 ����� 
    for j_1 = 1:size(k_combinations, 1)
        k = k_combinations(j_1, :)';
        k = [1; 1];
        gamma = -Inf;
        mu = -Inf;
        is_valid = true;
        for i_1 = 1:m
            
            H = A_all{i_1} - b_all{i_1}*k';
     
            % �������� �� ���������������� ���� ����������� ��������
            if ~all(abs(imag(eig(H))) < tol)
                is_valid = false;
                break;
            end
            % ��������, ��� ��� ����������� �������� ���������
            if length(unique(round(real(eig(H)), 10))) < length(eig(H))
                is_valid = false;
                break;
            end
            
             % �������� ������������ H
            if any(real(eig(H)) >= 0)
                is_valid = false;
                break;
            end

            if gamma < max(eig(H))
                gamma = max(eig(H));
            end
            
            [T, D] = eig(H);
            row_sum_T = sum(T, 2);
            row_sum_inv_T = sum(abs(pinv(T)), 2);
            mu_i = max(row_sum_inv_T) * max(row_sum_T);
            
            if mu_i > mu
                mu = mu_i;
            end
        end
        gamma = abs(gamma);
        
        if is_valid
            if (theta_2 == 0) || ((2*log(mu)/gamma) < theta_2)
                theta_2 = 2*log(mu)/gamma;
                k_2 = k;
            end
        end
    end
    if ~isempty(k_2)
        disp(['����������� k_2: ', mat2str(k_2')]);
        disp(['����������� 2*log(gamma)/mu: ', num2str(theta_2)]);
    else
        disp('���������� k �� �������');
    end
end

%������� ������� A_1 [-2 1; 0 -3]
%������� ������� b_1 [0; 1]
%������� ������� A_2 [-1 1; 0 -2]
%������� ������� b_2 [0; 1]
%������� ������� A_3 [-3 0; 1 -1]
%������� ������� b_3 [1; 0]
%R = eye(2);