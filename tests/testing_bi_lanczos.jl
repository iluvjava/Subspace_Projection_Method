include("test_utilities.jl")
include("../src/iterative_bi_lanczos.jl")

@testset begin 

    function Test1(n=5)
        @info "Testing Basics"
        A = rand(n, n)
        ibl = IterativeBiLanczos(A)
        V, W, T = ibl(n - 1)
        println("These are: V, W, T")
        display(V)
        display(W)
        display(T)
        println("Orthogonality between W,V is: ") 
        display(W'*V)
        orthogonalError = norm(W'*V - I, Inf)
        println("Testing Factorizations W'*A*V ≈ T")
        factorization = W'*A*V
        display(factorization)
        factError = norm(factorization - T, Inf)
        if orthogonalError > 1e-10
            @warn "The orthogonalization error for bi-lanczos is: $(orthogonalError), which is kinda big, please 
            check your code."

        elseif factError > 1e-10
            @warn "The factorizations error is $(factError), which is big, check your code. "
        end
        return factError < 1e-10 && orthogonalError < 1e-10
    end

    @info "Testing basics of Bi-Lanczos on a 5 by 5 real matrix with float64"
    @test Test1()

end