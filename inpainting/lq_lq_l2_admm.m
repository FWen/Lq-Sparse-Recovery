function [x1,x2,out] = lq_lq_l2_admm(A1,A2,y,mu,q1,q2,x01,x02,xtrue);
% lq_lq_l2_admm solves
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

rho  = 6;
rho1 = 1;
%rho2 = rho1;

if(isobject(A1))
    n1 = A1.n;
else
    n1 = size(A1,2);
end

if(isobject(A2))
    n2 = A2.n;
else
    n2 = size(A2,2);
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

z1 = zeros(n1,L);
z2 = zeros(n2,L);
w1 = zeros(n1,L);
w2 = zeros(n2,L);

beta = 1e-7;
beta0 = 1e3;%0.02*norm(y); 

out.et = [];out.e = [];
tic;

for iter = 1:MAX_ITER

    x1m1 = x1; x2m1 = x2;
    z1m1 = z1; z2m1 = z2;
    
    if beta0>beta % for acceleration of the algorithm
        beta0 = beta0 * 0.97;
    end
	
    if rho1<rho % for acceleration of the algorithm
        rho1 = rho1 * 1.01;
    end
    rho2 = rho1;
    
   % z-update      
    b1 = x1 + w1/rho1;
    b2 = x2 + w2/rho2;
    z1 = repmat(prox_Lq_m(b1, q1, rho1/mu/beta0),[1,L]) .* b1;
    z2 = repmat(prox_Lq_m(b2, q2, rho2/beta0),[1,L]) .* b2; 
    
    % x-update 
    %x1 = 1/(2*A1'*A1+rho*eye(n1)) * (2*(A1'*(y-A2*x2)) + rho*z1 - w1);
    %x2 = 1/(2*A2'*A2+rho*eye(n1)) * (2*(A2'*(y-A1*x1)) + rho*z2 - w2);
    % for orthonorm A1 and A2, the matrix inverse can be avoid
    x1 = 1/(2+rho1) * (2*(A1'*(y-A2*x2)) + rho1*z1 - w1);
    x2 = 1/(2+rho2) * (2*(A2'*(y-A1*x1)) + rho2*z2 - w2);

 	% w-update
 	w1 = w1 + rho1*(x1 - z1);
    w2 = w2 + rho2*(x2 - z2);
    
    if ~isquiet
        %out.e  = [out.e norm(x1-xtrue)/norm(xtrue)];
        out.e  = [out.e norm(x1m1-x1)/norm(xtrue)];
        out.et = [out.et, toc];
    end
        
    %Check for convergence: when both primal and dual residuals become small
    if (norm(rho1*(z1-z1m1))< sqrt(n1)*ABSTOL) & (norm(rho2*(z2-z2m1))< sqrt(n2)*ABSTOL)...
        & (norm(x1-z1) < sqrt(n1)*ABSTOL) & (norm(x2-z2) < sqrt(n2)*ABSTOL) 
        break;           
    end    
    
end

end
