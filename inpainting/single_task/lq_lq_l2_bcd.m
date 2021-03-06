function [x1,x2,out] = lq_lq_l2_bcd(A1,A2,y,mu,q1,q2,x01,x02,xtrue);
% lq_lq_l2_bcd solves
%
%   minimize 1/\beta*|| A1*x1 +A2*x2 - y ||_2^2 + \mu*|| x1 ||_q1^q1 + || x2 ||_q2^q2
%
% Inputs
%	A1,A2,y,mu: 
%	1=>q1,q2=>0
%	xtrue: for debug, for calculation of errors
%   x1,x2: intialization
% Outputs
%	x1,x2: the recovery
%	out.e: the error with respect to the true
%	out.et: time index

%Convergence setup
MAX_ITER = 300;
ABSTOL = 1e-6;

[m, L] = size(y);

if(isobject(A1))
    n1 = A1.n;
    L1 = 2*A1.eig;
else
    n1 = size(A1,2);
    L1 = 2*norm(A1'*A1);
end

if(isobject(A2))
    n2 = A2.n;
    L2 = 2*A2.eig;
else
    n2 = size(A2,2);
    L2 = 2*norm(A2'*A2);
end

%Initialize
if nargin<7
	x1 = zeros(n1,L);
    x2 = zeros(n2,L);
else
    x1 = x01;
    x2 = x02;
end

isquiet = 0;
if nargin<8
    isquiet = 1;
end

beta = 1e-7;
beta0 = 1e3;%0.02*norm(y); 

out.et = [];out.e = [];
tic;
for iter = 1 : MAX_ITER

    if beta0>beta
        beta0 = beta0 * 0.97;
    end

    x1m1 = x1;
    x2m1 = x2;

    c1 = x1 - 2/L1*(A1'*(A1*x1 + A2*x2 - y));
    %x1 = repmat(prox_Lq_m(c1,q1,L1/beta0/mu),[1,L]) .* c1;   % multitask
    x1 = [prox_Lq(c1(:,1),q1,L1/beta0/mu), prox_Lq(c1(:,2),q1,L1/beta0/mu), prox_Lq(c1(:,3),q1,L1/beta0/mu)]; % single-task
    
    c2 = x2 - 2/L2*(A2'*(A1*x1 + A2*x2 - y));
    %x2 = repmat(prox_Lq_m(c2,q2,L2/beta0),[1,L]) .* c2;   % multitask
    x2 = [prox_Lq(c2(:,1),q2,L2/beta0), prox_Lq(c2(:,2),q2,L2/beta0), prox_Lq(c2(:,3),q2,L2/beta0)]; % single-task

    
    if ~isquiet
        out.e  = [out.e norm(x1-xtrue)/norm(xtrue)];
        out.et = [out.et toc];
    end
    
    %Check for convergence
    if norm(x1-x1m1)<ABSTOL*sqrt(n1) & norm(x2-x2m1)<ABSTOL*sqrt(n2)
        break;
    end

end
