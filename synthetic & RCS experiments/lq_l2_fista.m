function [x,out] = lq_l2_fista(A,y,lamda,q1,q2,N,L,xtrue,x0);
% lq_l2_fista solves
%
%   minimize || Ax - y ||_2^2 + \lambda || x ||_q^q
%
% Inputs
%	A,y,lambda: CS variables
%	Q=[q_1,...,q_M], 1=>q_1>...>q_M=>0
%	xtrue: for debug, for calculation of errors
%   x0: intialization
% Outputs
%	x: the CS recovery
%	out.e: the error with respect to the true
%	out.et: time index

Aty = A'*y;
if(isobject(A))
    m=A.m;
    n=A.n;
else
    [m,n]=size(A);
    %A2  = A'*A;
end

%Compute Lipschitz constant
% L = 2.1*norm(A'*A);
% L = 2.0;   %for orthogonalized A

%Convergence setup
MAX_ITER = 6e2;
ABSTOL   = 1e-6;

%Initialize
if nargin<9
	x = zeros(n,1);
else
    x = x0;
end
t = 1;
u = x;

lamda0 = 0.1*norm(y);

out.et = [];out.e = [];
tic;

for iter = 1:MAX_ITER
    if lamda0>lamda % for acceleration of the algorithm
        lamda0 = lamda0 * 0.97;
    end
    
    xm1 = x;	

    %v = u - (1/L)*(A2*u - Aty);
    v = u - (1/L)*(A'*(A*u) - Aty);

    %x = shrinkage_Lq(v, q1, lamda0, L); 
    x(1:N,1)   = shrinkage_Lq(v(1:N), q1, lamda0, L); 
    x(N+1:n,1) = shrinkage_Lq(v(N+1:end), q2, lamda0, L); 

    tp1 = (1 + sqrt(1+4*t^2))/2;

    u = x + (t-1)/(tp1)*(x-xm1);

    t = tp1;	


    out.e  = [out.e norm(x-xtrue)/norm(xtrue)];
    out.et = [out.et toc];

    if iter>200 && norm(x-xm1)<ABSTOL*sqrt(n)
          break;
    end

end


