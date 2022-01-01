    n = 1024;
    A = spdiags(linspace(1e-3, 1, n)'.^3, 0, n, n); 
    b = randn(n, 1);
    Times = [];
%% 
for II = 1: 30
    tic
        pcg(A, b, 1e-10, 1024^2); 
    Times(end + 1) = toc;
end

