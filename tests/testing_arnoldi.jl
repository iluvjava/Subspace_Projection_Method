include("test_utilities.jl")
include("../src/arnoldi.jl")


@testset "Basic Tests for Iterative Hessenberg" begin

    function Test1(N=10)
        @info "Testing basic full Hessenberg, direct run no checking: "
        A = rand(N, N)
        ih = Arnoldi(A, rand(N, 1))
        for II = 1:N
            ih() 
        end
        display(GetHessenberMatrix(ih))
        display(GetOrthogonalMatrix(ih))
        return true
    end
    
    function Test2(N=10)
        @info "Checking the errors on the Hessenberg decomposition: "
        A = rand(N, N)
        ih = Arnoldi(A, rand(N, 1))
        for II = 1:N-2
            ih() 
        end
        
        H = GetHessenberMatrix(ih)
        print("Matrix H: ")
        display(H)
        Q = GetOrthogonalMatrix(ih)
        print("Matrix Q:")
        display(Q)
    
        @assert (A*Q[:, 1:end-1] - Q*H).|>abs|>sum|>mean < 1e-10 "Decomposition Errors too big"
        return true
    end
    
    function Test3(N=10)
        @info "Checking Tridiagonal Decomposition on Hermitian Matrices: "
        A = rand(N, N)
        A = A*A'
        ih = Arnoldi(A, rand(N, 1))
        for II = 1:N-2
            ih() 
        end
    
        H = GetHessenberMatrix(ih)
        println("H Matrix:")
        display(H)
        Q = GetOrthogonalMatrix(ih)
        println("Q Matrix")
        display(Q)
        @info "Assertring Error bounds of decomposition and Tridiagonal Property of the H Matrix"
        @assert (H - tril(H, 1)).|>abs|>sum|>mean < 1e-10 "Not Tridiagonal Enough"
        @assert (A*Q[:, 1:end-1] - Q*H).|>abs|>sum|>mean < 1e-10 "Decomposition Error too big"
        return true
    end
    
    function Test4(N=10)
        @info "Testing Limited Memory Arnoldi Decomposition"
        A = rand(N, N)
        ih = Arnoldi(A, rand(N, 1); max_k=2)
        for II = 1:N-2
            ih()
        end
        H = GetHessenberMatrix(ih)
        println("Limited Memory H matrix: ")
        display(H)
        return true
    
    end
    
    function Test5(N=10)
        @info "Testing Limited Memory Arnoldi Decomposition with a complex matrix"
        A = rand(N, N) + rand(N, N)im
        ih = Arnoldi(A, rand(N, 1)im; max_k=2)
        for II = 1:N-2
            ih()
        end
        H = GetHessenberMatrix(ih)
        println("Limited Memory H matrix: ")
        display(H)
        Q = GetOrthogonalMatrix(ih)
        println("Limited Memory Q Matrix")
        display(Q)
        return true
    
    end
    
    @test Test1()
    @test Test2()
    @test Test3()
    @test Test4()
    @test Test5()
end
