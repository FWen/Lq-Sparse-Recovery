clear all; clc; close all;

image_file = 'turtle.png';

XX = imread(image_file);
HIGHT = size(XX,1);
WIDTH = size(XX,2);
n = WIDTH*HIGHT;   	% signal dimension

figure(1);subplot(3,4,1);imshow(XX);set(gcf,'outerposition',get(0,'screensize'));title('Original');

A1 = mtxIDCT(n); 
A2 = mtxEYE(n);

X = double([reshape(XX(:,:,1),[n,1]), reshape(XX(:,:,2),[n,1]), reshape(XX(:,:,3),[n,1])]);
x = A1'*X;%true dct coefficients

y = X;   
J = randperm(n); 
J = J(1:round(0.3*n));   %corruption ratio 
y(J,:) = 255;

figure(1);subplot(3,4,1);imshow(v_2_color_imag(y, HIGHT, WIDTH));
title(sprintf('Corrupted\n RelErr=%.3f, PSNR=%.2f dB',norm(y-X,'fro')/norm(X,'fro'),psnr(y, X)));
t0 = tic;


% JP-----------------------
x_JP = YALL1_admm(A1, y, 1,zeros(size(x)), x);
RelErr(1) = norm(x_JP-x,'fro')/norm(x,'fro');

figure(1);subplot(3,4,3);imshow(v_2_color_imag(idct(x_JP), HIGHT, WIDTH));
title(sprintf('JP\n RelErr=%.3f, PSNR=%.2f dB',RelErr(1),psnr(idct(x_JP), X)));

sprintf('JP completed, elapsed time: %.1f seconds',toc(t0))
t0=tic;



% YALL1--------------------
n_mu = 15;
mu   = logspace(-1.5, 1,n_mu);
relerr_yal = zeros(1,n_mu);
xx_yal = zeros(n,3,n_mu);   
for k = 1:length(mu); 
    xr = YALL1_admm(A1, y, mu(k), zeros(size(x)), x);
    relerr_yal(k)  = norm(xr-x,'fro')/norm(x,'fro');
    xx_yal(:,:,k) = xr;
end
[RelErr(2) mi] = min(relerr_yal);
x_YALL1 = xx_yal(:,:,mi); 

figure(1);subplot(3,4,4);imshow(v_2_color_imag(idct(x_YALL1), HIGHT, WIDTH));
title(sprintf('YALL1\n RelErr=%.3f, PSNR=%.2f dB',RelErr(2),psnr(idct(x_YALL1), X)));

sprintf('YALL1 completed, elapsed time: %.1f seconds',toc(t0))



% Lq-Lq-BCD and Lq-Lq-ADMM--------------------
% Initialization. S-ADMM (with q1=q2=1 and \mu=1) and JP solves the same convex formulation, 
% and have the same accuracy
x01 = x_JP;
x02 = y - A1*x_JP;

qs = 0:0.1:1;
relerr_bcd = zeros(1,n_mu); relerr_admm = relerr_bcd;
xx_bcd  = zeros(n,3,n_mu); xx_admm = xx_bcd;
x_bcd   = zeros(n*3,length(qs),length(qs)); x_admm = x_bcd;
PSNR_bcd = zeros(length(qs)); PSNR_admm = PSNR_bcd; 
RelErrs_bcd = zeros(length(qs)); RelErrs_admm = RelErrs_bcd;

for l1=1:length(qs)
    for l2=1:length(qs)
        
        t0=tic;
        for k = 1:length(mu)
            [xr1,~,out] = lq_lq_l2_bcd(A1, A2, y, mu(k), qs(l1), qs(l2), x01, x02, x);
            relerr_bcd(k) = norm(xr1-x,'fro')/norm(x,'fro');
            xx_bcd(:,:,k) = xr1;
            
            [xr2] = lq_lq_l2_admm(A1, A2, y, mu(k), qs(l1), qs(l2), x01, x02, x);
            relerr_admm(k) = norm(xr2-x,'fro')/norm(x,'fro');
            xx_admm(:,:,k) = xr2;
        end
        [RelErrs_bcd(l1,l2), mi] = min(relerr_bcd); 
        x_bcd(:,l1,l2)  = reshape(xx_bcd(:,:,mi),[n*3,1]); 
        PSNR_bcd(l1,l2) = psnr(idct(reshape(x_bcd(:,l1,l2),[n,3])), X);
        
        [RelErrs_admm(l1,l2), mi] = min(relerr_admm); 
        x_admm(:,l1,l2)  = reshape(xx_admm(:,:,mi),[n*3,1]); 
        PSNR_admm(l1,l2) = psnr(idct(reshape(x_admm(:,l1,l2),[n,3])), X);
        
        sprintf('BCD and ADMM with q1=%.1f and q2=%.1f completed, elapsed time: %.1f seconds',qs(l1),qs(l2),toc(t0))
        
    end
end

qi = [3, 6, 8];
for k=1:length(qi)
    %BCD with q1=q2=qs(qi(k))
    figure(1);subplot(3,4,4+k);
    imshow(v_2_color_imag(idct(reshape(x_bcd(:,qi(k),qi(k)),[n,3])), HIGHT, WIDTH));
    title(sprintf('BCD (q1=q2=%.1f)\n RelErr=%.3f, PSNR=%.2f dB',qs(qi(k)),RelErrs_bcd(qi(k),qi(k)),PSNR_bcd(qi(k),qi(k))));

     %ADMM with q1=q2=qs(qi(k))
    figure(1);subplot(3,4,8+k);
    imshow(v_2_color_imag(idct(reshape(x_admm(:,qi(k),qi(k)),[n,3])), HIGHT, WIDTH));
    title(sprintf('ADMM (q1=q2=%.1f)\n RelErr=%.3f, PSNR=%.2f dB',qs(qi(k)),RelErrs_admm(qi(k),qi(k)),PSNR_admm(qi(k),qi(k))));
end


[w1, e1] = max(PSNR_bcd);[~, lo] = max(w1); ko = e1(lo);
figure(1);subplot(3,4,8);
imshow(v_2_color_imag(idct(reshape(x_bcd(:,ko,lo),[n,3])), HIGHT, WIDTH));
title(sprintf('BCD (q1=%.1f, q2=%.1f)\n RelErr=%.3f, PSNR=%.2f dB',qs(ko),qs(lo),RelErrs_bcd(ko,lo),PSNR_bcd(ko,lo)));

[w1, e1] = max(PSNR_admm);[~, lo] = max(w1); ko = e1(lo);
figure(1);subplot(3,4,12);
imshow(v_2_color_imag(idct(reshape(x_admm(:,ko,lo),[n,3])), HIGHT, WIDTH));
title(sprintf('ADMM (q1=%.1f, q2=%.1f)\n RelErr=%.3f, PSNR=%.2f dB',qs(ko),qs(lo),RelErrs_admm(ko,lo),PSNR_admm(ko,lo)));

v0 = min(min(PSNR_bcd)); v1 = PSNR_bcd(ko,lo);
figure(5);subplot(121);
contourf(qs,qs,PSNR_bcd,[v0:2:v1]);    colorbar; xlabel('q_2');ylabel('q_1');
set(gca, 'CLim', [v0, v1]);

v0 = min(min(PSNR_admm)); v1 = PSNR_admm(ko,lo);
figure(5);subplot(122);
contourf(qs,qs,PSNR_admm,[v0:2:v1]);   colorbar; xlabel('q_2');ylabel('q_1');
set(gca, 'CLim', [v0, v1]);

