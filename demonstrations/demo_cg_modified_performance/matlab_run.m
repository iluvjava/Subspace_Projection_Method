
    n = 512;
    A = spdiags(rand(n, 1).^6, 0, n, n); 
    b = randn(n, 1);
tic
    pcg(A, b, 1e-10, n^3); 
toc

