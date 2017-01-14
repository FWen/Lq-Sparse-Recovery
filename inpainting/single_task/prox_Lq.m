function [x] = prox_Lq(b, q, phi)

if (q==0)
    x = b;
    i1 = find(abs(b)<=sqrt(2/phi));
    x(i1) = 0;
elseif (q<1 && q>0)
    max_iter = 10;
    ABSTOL   = 1e-5;
    x    = zeros(length(b),1);
    ab   = abs(b);
    beta = ( 2*(1-q)/phi )^(1/(2-q));
    tao  = beta + q*beta^(q-1)/phi;
    i0   = find(ab>tao);
    if length(i0)>0 
        b_u = ab(i0);
        x_u = b_u;
        for k=1:max_iter              
            deta_x = (q*x_u.^(q-1) + phi*x_u - phi*b_u) ./ (q*(q-1)*x_u.^(q-2) + phi);
            x_u    = x_u - deta_x;
            if (k>2 && norm(deta_x) < sqrt(length(x_u))*ABSTOL )
                break;
            end
        end
        x_u = x_u .* sign(b(i0));
        x(i0) = x_u;
    end 
elseif (q==1)
    x = sign(b) .* max(abs(b)-1/phi, 0);
end

end
