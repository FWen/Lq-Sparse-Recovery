clear all; clc;

N = 128;
M = 128;

A1 = dctmtx(N);

A2 = randn(M,N);
A2 = orth(A2')'; 

Ks = [1, 2:2:46];

mu = logspace(-2, 1,12);

num_MC = 100;

for nK=1: length(Ks)
    
    for l = 1:num_MC
        disp(['Sparsity: ', num2str(Ks(nK)),'  Time: ', num2str(l)]);
        t0 = tic;
        
        x1 = 5*SparseVector(N,Ks(nK));
        x2 = 5*SparseVector(N,Ks(nK));

        noise = randn(M,1);
        y  = A1*x1 + A2*x2 + 0.001*noise/std(noise);

                
        % S-ADMM with q1=q2=1 and \mu=1, for initialization
        [x01,x02,~] = lq_lq_admm(A1, A2, y, 1, 1, 1, zeros(N,1), zeros(N,1), x1);
        norm([x01;x02]-[x1;x2])/norm([x1;x2]);

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
            
            
            % FISTA
            L=2.1*(1/mu(k)/mu(k)+1);
            [xr,~] = lq_l2_fista([A1/mu(k),A2],y,1e-6,0.7,0.7,N,L,[mu(k)*x1;x2],[mu(k)*x01;x02]);
            relerrs(8,k) = norm(xr(1:N,1)/mu(k)-x1)/norm(x1);
            
            [xr,~] = lq_l2_fista([A1/mu(k),A2],y,1e-6,0.5,0.5,N,L,[mu(k)*x1;x2],[mu(k)*x01;x02]);
            relerrs(9,k) = norm(xr(1:N,1)/mu(k)-x1)/norm(x1);
            
            [xr,~] = lq_l2_fista([A1/mu(k),A2],y,1e-6,0.5,0.5,N,L,[mu(k)*x1;x2],[mu(k)*x01;x02]);
            relerrs(10,k) = norm(xr(1:N,1)/mu(k)-x1)/norm(x1);
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
    Ks,Success_Rate(:,5),'b-',Ks,Success_Rate(:,6),'b--',Ks,Success_Rate(:,7),'b:+',...
    Ks,Success_Rate(:,8),'g-',Ks,Success_Rate(:,9),'g--',Ks,Success_Rate(:,10),'g:+','linewidth',1);grid off;
legend('S-ADMM (q=1)','BCD (q=0.7)','BCD (q=0.5)','BCD (q=0.2)','ADMM (q=0.7)','ADMM (q=0.5)','ADMM (q=0.2)',...
        'FISTA (q=0.7)','FISTA (q=0.5)','FISTA (q=0.2)','Location','SouthWest');
ylabel('Frequency of success'); xlabel('Sparsity K');
xlim([0 Ks(end)]);
