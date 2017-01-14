clear all; clc;

N = 256;
M = 100;

A1 = dctmtx(N);
J = randperm(N); 
A1 = A1(J(1:M),:);

A2 = eye(M);

Ks = [1, 2:2:46];

mu = logspace(-2, 1,12);
num_MC = 100;

for nK=1: length(Ks)
    
    for l = 1:num_MC
        disp(['Sparsity: ', num2str(Ks(nK)),'  Time: ', num2str(l)]);
        t0 = tic;
        
        x1 = 5*SparseVector(N,Ks(nK));
        x2 = stblrnd(1, 0, 1e-3, 0, M, 1);
        
        y  = A1*x1 + x2;
%         SNR = 20*log10(norm(A1*x1)/norm(x2))

        
        % S-ADMM with q1=q2=1 and \mu=1, for initialization
        [x01,x02,~] = lq_lq_admm(A1, A2, y, 1, 1, 1, zeros(N,1), zeros(M,1), x1);

        for k = 1:length(mu)
            % ADMM, q1=q2=1
            [xr1,xr2,~] = lq_lq_admm(A1, A2, y, mu(k), 1, 1, x01, x02, x1);
            relerrs(1,k) = norm(xr1-x1)/norm(x1);
            
            % Proposed BCD
            [xr1,xr2,~] = lq_lq_l2_bcd(A1, A2, y, mu(k), 0.7, 0.7, x01, x02, x1);
            relerrs(2,k) = norm(xr1-x1)/norm(x1);
            
            [xr1,xr2,~] = lq_lq_l2_bcd(A1, A2, y, mu(k), 0.5, 0.5, x01, x02, x1);
            relerrs(3,k) = norm(xr1-x1)/norm(x1);

            [xr1,xr2,~] = lq_lq_l2_bcd(A1, A2, y, mu(k), 0.2, 0.2, x01, x02, x1);
            relerrs(4,k) = norm(xr1-x1)/norm(x1);

            
            % Proposed ADMM
            [xr1,xr2,~] = lq_lq_l2_admm(A1, A2, y, mu(k), 0.7, 0.7, x01, x02, x1);
            relerrs(5,k) = norm(xr1-x1)/norm(x1);
            
            [xr1,xr2,~] = lq_lq_l2_admm(A1, A2, y, mu(k), 0.5, 0.5, x01, x02, x1);
            relerrs(6,k) = norm(xr1-x1)/norm(x1);

            [xr1,xr2,~] = lq_lq_l2_admm(A1, A2, y, mu(k), 0.2, 0.2, x01, x02, x1);
            relerrs(7,k) = norm(xr1-x1)/norm(x1);
            
        end
        
        RelErr(l,:) = min(relerrs')';
        toc(t0)

    end
    aver_Err(nK,:) = mean(RelErr);
        
    % successful rate of recovery
    for m=1:size(RelErr,2)
        idx_suc = find(RelErr(:,m)<=1e-2);
        Success_Rate(nK,m) = length(idx_suc)/num_MC;
    end
    
end


figure(1);
plot(Ks,Success_Rate(:,1),'k-',Ks,Success_Rate(:,2),'r-',Ks,Success_Rate(:,3),'r--',Ks,Success_Rate(:,4),'r:+',...
    Ks,Success_Rate(:,5),'b-',Ks,Success_Rate(:,6),'b--',Ks,Success_Rate(:,7),'b:+','linewidth',1);grid off;
legend('YALL1','BCD (q=0.7)','BCD (q=0.5)','BCD (q=0.2)','ADMM (q=0.7)','ADMM (q=0.5)','ADMM (q=0.2)',1);
ylabel('Frequency of success'); xlabel('Sparsity K');
xlim([0 Ks(end)]);
