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

    function Test2(n=5)
        @info "Testing it with a ($n, $n) Complex Hermitian Matrix"
        A = rand(n, n) + rand(n, n)*im
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
        return true
    end

    function Test3(n=5)
        @info "Testing it with a ($n, $n) Exact Hermitian Matrix"
        A = rand(n, n)
        A = A' + A
        A = convert(Matrix{Rational{BigInt}}, A)
        b = convert(Vector{Rational{BigInt}}, rand(n))
        il = IterativeLanczos(A, b)
        il(n - 1)
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

    """
    Testing the fully re-orthgonalization functionality of the algorithm. 
    """
    function test4(n=128)
        @info "testin fully re-orthgonalization"
        A = rand(n, n)
        A = A' + A
        A = convert(Matrix{Rational{BigInt}}, A)
        b = convert(Vector{Rational{BigInt}}, rand(n))
        il = IterativeLanczos(A, b)
        il.reorthogonalize = true
        il(n - 1)
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
    @test Test2()
    @test Test3()
    @test test4()

end