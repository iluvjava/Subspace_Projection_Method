include("../src/iterative_lanczos.jl")
using LinearAlgebra 
using SparseArrays
using Test

@testset begin 
    function Test1(n=5)
        @info "Testing Iterative Lanczos on a ($n,$n) matrix. "
        A = rand(n, n)
        A = A' + A
        b = rand(n)
        il = IterativeLanczos(A, b)
        betas = il(n - 1)
        T = GetTMatrix(il)
        Q = GetQMatrix(il)
        println("The T matrix is: ")
        display(T)
        println("The Q matrix is: ")
        display(Q)
        println("The Q'AQ is: ")
        QAQ = Q'*A*Q
        display(QAQ)
        error = norm(QAQ - T, Inf)
        if error >= 1e-10
            @warn "The inf error of Q'AQ - T is $error, which is lager than expected, please check the code. "
        end
        return error < 1e-10
    end
    @test Test1()

end