# INVESTIGATE THE FOLLOWING: 
# * Orthogonalize the conjugate vector and the residual vector together, periodically against previous vectors, does it 
#   reduce the number of iterations? 


mutable struct ConjGradModified{T <: Number}
    A::Function              # Linear opeartor
    b::AbstractArray{T}      # RHS vector
    tensor_size::Tuple       # The size of the tensor the linear operator is acting on. 

    x::Vector{T}        # current guess, started with initial guess. 
    r::Vector{T}        # previous computed residual
    rnew::Vector{T}     # Current residual
    d::Vector{T}        # Conjugate Direction
    Ad::Vector{T}       # for reducing garbage collector time. 
    
    itr::UInt64              # Iteration count. 
    Q::Vector{Vector{T}}     # list of normalized residuals vectors. 
    Ds::Vector{Vector{T}}    # ist of normalized Ad vector during the iterations. 

    
    function ConjGradModified(
        A::Function, 
        b::AbstractArray,
        x0::Union{AbstractArray,Nothing}=nothing
    )
        x0 = x0===nothing ? b .+ 0.1 : x0
        r = reshape(b - A(x0), :)
        T = typeof(r[1])
        this = new{T}()
        this.r = r
        this.A = A
        this.b = b
        this.tensor_size = size(b)
        this.x = x0
        this.rnew = similar(this.r)
        this.d = this.r
        this.Ad = similar(this.r)
        ComputeVec!(this, this.r, this.Ad)
        this.x = x0
        this.itr = 0

        this.Q = Vector{Vector{T}}()
        this.Ds = Vector{Vector{T}}()
        push!(this.Q, this.r/norm(this.r))
        push!(this.Ds, this.d/norm(this.d))
        return this
    end


    function ConjGradModified(
            A::AbstractMatrix, 
            b::AbstractVector, 
            x0::Union{AbstractArray,Nothing}=nothing
        )
        return ConjGradModified((x)-> A*x, b, x0)
    end

end


"""
    Apply the linear transformation on the tensor and return the 
    residual vector after transformation. 
"""
function ComputeResidualVec(this::ConjGradModified, x::AbstractArray)
    resVec = this.b - this.A(reshape(x, this.tensor_size))
    return reshape(resVec, length(resVec))
end


"""
    Get the norm of the current residual of Modified Conjugate 
    Gradient. 
"""
function GetResidualNorm(this::ConjGradModified)
    return norm(this.r)
end


"""
    Compute the vector using the linear transformation. 
"""
function ComputeVec(this::ConjGradModified, x::AbstractArray)
    vec = this.A(reshape(x, this.tensor_size))
    return reshape(vec, length(vec))
end


"""
    Pre-allocates a vector and get the values to the given 
    pre-allocated vector. 
"""
function ComputeVec!(this::ConjGradModified, x::AbstractArray, vec::AbstractArray)
    vec .= reshape(this.A(reshape(x, this.tensor_size)), :)
    return nothing
end


"""
    Force the orthogonalization happen between the residual vectors against 
    all previous residual vectors, and the conjugate vector as well. 
"""
function ForceOrthogonalize(this::ConjGradModified)
    for Idx in 1: length(this.Q)
        for Jdx in 1: Idx - 1
            # this.Q[Idx] .-= dot(this.Q[Jdx], this.Q[Idx])*this.Q[Jdx]
            this.Ds[Idx] .-= dot(this.Ds[Jdx], this.Ds[Idx])*this.Ds[Jdx]
        end
    end
    for  Aq in this.Q
        this.d .-= dot(Aq, this.d)*Aq
    end

    return
end


"""
    Performs One step of conjugate Gradient. 
"""
function (this::ConjGradModified)()
    r = this.r
    if norm(r) == 0
        return 0 # The problem is solved already. 
    end
    d = this.d
    Ad = this.Ad
    a = dot(r, r)/dot(d, Ad)
    
    
    if a < 0 
        error("CG got a non-definite matrix")
    end

    this.x += a*d
    this.rnew .= r - a*Ad                    # update rnew 
    rnewNorm = norm(this.rnew)
    
    b = dot(this.rnew, this.rnew)/dot(r, r)
    # @assert abs(dot(rnew + β*d, Ad)) < 1e-8 "Not conjugate"
    this.d = this.rnew + b*d
    this.r = copy(this.rnew)                      # Override
    this.itr += 1 
    ComputeVec!(this, this.d, Ad)
    return convert(Float64, rnewNorm)
end


"""
    Performs j iterations of conjugate gradient, collect the 
    2norm of the residual and return all of them in a vector. 
"""
function (this::ConjGradModified)(j::Int64)
    results = Vector()
    for _ in 1: j
        push!(results, this())
    end
    return results
end


### Using stuff. 
using Test, LinearAlgebra
@testset begin
    
    function Test1()
        n = 10
        A = rand(n, n)
        A = A'*A
        display(A)
        b = rand(n)
        cg = ConjGradModified(A, b)
        for I in 1:n + 1
            println("Iter: $I; "*"$(cg())")
        end
        return norm(b - A*cg.x) < 1e-8

    end
    function Test2()
        n = 5
        A = rand(n, n)
        A = A'*A
        display(A)
        b = rand(n)
        cg = ConjGradModified(A, b)
        for I in 1:n + 1
            println("Iter: $I; "*"$(cg())")
            ForceOrthogonalize(cg)
        end
        return norm(b - A*cg.x) < 1e-8
    end

    @test Test1()
    @test Test2()
end