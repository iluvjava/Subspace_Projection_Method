tic
    n = 128^2; 
    A = spdiags(rand(n, 1), 0, n, n); 
    b = randn(n, 1);
    pcg(A, b, 1e-2, n);
    
toc

