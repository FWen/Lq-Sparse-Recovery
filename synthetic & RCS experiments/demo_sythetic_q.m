clear all; clc;

N = 128;
M = 128;
K = 25;

A1 = dctmtx(N);

A2 = randn(M,N);
A2 = orth(A2')'; 

x1 = 5*SparseVector(N,K);
x2 = 5*SparseVector(N,K);

noise = randn(M,1);
y  = A1*x1 + A2*x2 + 0.001*noise/std(noise);
        
mu = logspace(-1, 1,15);

% S-ADMM with q1=q2=1 and \mu=1, for initialization
[x01,x02,~] = lq_lq_admm(A1, A2, y, 1, 1, 1, zeros(N,1), zeros(N,1), x1);
norm([x01;x02]-[x1;x2])/norm([x1;x2])
        

% Lq-Lq-BCD and Lq-Lq-ADMM--------------------
qs = 0:0.1:1;

for l1=1:length(qs)
    for l2=1:length(qs)
        for k = 1:length(mu)
           
            [xr1,xr2,~] = lq_lq_l2_bcd(A1, A2, y, mu(k), qs(l1), qs(l2), x01, x02, x1);
            relerr_bcd(k) = norm(xr1-x1)/norm(x1);
            
            [xr1,xr2,~] = lq_lq_l2_admm(A1, A2, y, mu(k), qs(l1), qs(l2), x01, x02, x1);
            relerr_admm(k) = norm(xr1-x1)/norm(x1);

            L=2.1*(1/mu(k)/mu(k)+1);
            [xr,~] = lq_l2_fista([A1/mu(k),A2],y,1e-6,qs(l1),qs(l2),N,L,[mu(k)*x1;x2],[mu(k)*x01;x02]);
            relerr_fista(k) = norm(xr(1:N,1)/mu(k)-x1)/norm(x1);
        end
        
        [RelErrs_bcd(l1,l2), mi] = min(relerr_bcd);       
        [RelErrs_admm(l1,l2), mi] = min(relerr_admm);        
        [RelErrs_fista(l1,l2), mi] = min(relerr_fista); 

        sprintf('q1=%.1f and q2=%.1f completed',qs(l1),qs(l2))
    end
end

figure(6);
subplot(131);
contourf(qs,qs,20*log10(RelErrs_fista));    
colorbar; xlabel('q_2');ylabel('q_1');title('FISTA');
subplot(132);
contourf(qs,qs,20*log10(RelErrs_bcd));  
colorbar; xlabel('q_2');ylabel('q_1');title('BCD');
subplot(133);
contourf(qs,qs,20*log10(RelErrs_admm));  
colorbar; xlabel('q_2');ylabel('q_1');title('ADMM');

