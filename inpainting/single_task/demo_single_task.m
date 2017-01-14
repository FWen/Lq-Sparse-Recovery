clear all; clc; close all;

image_file = 'input_transport.png';

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

figure(1);subplot(3,4,2);imshow(v_2_color_imag(y, HIGHT, WIDTH));
title(sprintf('Corrupted\n RelErr=%.3f, PSNR=%.2f dB',norm(y-X,'fro')/norm(X,'fro'),psnr(y, X)));
t0 = tic;


% JP-----------------------
x_JP = YALL1_admm(A1, y, 1, zeros(size(x)), x);
RelErr(1) = norm(x_JP-x,'fro')/norm(x,'fro');

figure(1);subplot(3,4,3);imshow(v_2_color_imag(idct(x_JP), HIGHT, WIDTH));
title(sprintf('JP\n RelErr=%.3f, PSNR=%.2f dB',RelErr(1),psnr(idct(x_JP), X)));

sprintf('JP completed, elapsed time: %.1f seconds',toc(t0))
t0=tic;


% YALL1--------------------
n_mu = 15;
mu     = logspace(-1.5, 1,n_mu);
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

qs  = [0.2,0.2; 0.5,0.5; 0.7,0.4]';
qs2 = [0.2,0.2; 0.5,0.5; 0.7,0.4]';


relerr_bcd = zeros(1,n_mu); relerr_admm = relerr_bcd;
xx_bcd  = zeros(n,3,n_mu); xx_admm = xx_bcd;
x_bcd   = zeros(n,3,size(qs,2)); x_admm = x_bcd;
PSNR_bcd = zeros(1,size(qs,2)); PSNR_admm = PSNR_bcd; 
RelErrs_bcd = zeros(1,size(qs,2)); RelErrs_admm = RelErrs_bcd;

for l=1:size(qs,2)
    t0=tic;
    
    for k = 1:length(mu)
        [xr1] = lq_lq_l2_bcd(A1, A2, y, mu(k), qs(1,l), qs(2,l), x01, x02, x);
        relerr_bcd(k) = norm(xr1-x,'fro')/norm(x,'fro');
        xx_bcd(:,:,k) = xr1;

        [xr2] = lq_lq_l2_admm(A1, A2, y, mu(k), qs2(1,l), qs2(2,l), x01, x02, x);
        relerr_admm(k) = norm(xr2-x,'fro')/norm(x,'fro');
        xx_admm(:,:,k) = xr2;
    end
    [RelErrs_bcd(l), mi] = min(relerr_bcd); 
    x_bcd(:,:,l)  = xx_bcd(:,:,mi); 
    PSNR_bcd(l) = psnr(idct(x_bcd(:,:,l)), X);

    [RelErrs_admm(l), mi] = min(relerr_admm); 
    x_admm(:,:,l)  = xx_admm(:,:,mi); 
    PSNR_admm(l) = psnr(idct(x_admm(:,:,l)), X);

    sprintf('BCD and ADMM with q1=%.1f and q2=%.1f completed, elapsed time: %.1f seconds',qs(1,l),qs(2,l),toc(t0))
end

for k=1:size(qs,2)
    %BCD with q1=qs(1,k),q2=qs(2,k)
    figure(1);subplot(3,4,4+k);
    imshow(v_2_color_imag(idct(x_bcd(:,:,k)), HIGHT, WIDTH));
    title(sprintf('BCD (q1=%.1f, q2=%.1f)\n RelErr=%.3f, PSNR=%.2f dB',qs(1,k),qs(2,k),RelErrs_bcd(k),PSNR_bcd(k)));

     %ADMM with q1=qs2(1,k),q2=qs2(2,k)
    figure(1);subplot(3,4,8+k);
    imshow(v_2_color_imag(idct(x_admm(:,:,k)), HIGHT, WIDTH));
    title(sprintf('ADMM (q1=%.1f, q2=%.1f)\n RelErr=%.3f, PSNR=%.2f dB',qs2(1,k),qs2(2,k),RelErrs_admm(k),PSNR_admm(k)));
end
