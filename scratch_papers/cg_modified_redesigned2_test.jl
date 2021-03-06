# We are going to make the conjugate vectors more conjugate. 

mutable struct ConjGradModified{T <: Number}
    A::Function              # Linear opeartor
    b::AbstractArray{T}         # RHS vector
    tensor_size::Tuple       # The size of the tensor the linear operator is acting on. 

    x::Vector{T}        # current guess, started with initial guess. 
    r::Vector{T}        # previous computed residual
    rnew::Vector{T}     # Current residual
    d::Vector{T}        # Conjugate Direction
    Ad::Vector{T}       # for reducing garbage collector time. 
    
    itr::UInt64              # Iteration count. 
    Q::Union{Nothing, Vector{Vector{T}}}       # Re-Orthogonalization basis.

    storage_limit::UInt64    # storage limit for the Q vector, default is n - 1
    reorthogonalize::Bool    # Whether to perform reorthogonalization.
    ortho_period::Int64

    function ConjGradModified(
        A::Function, 
        b::AbstractArray, 
        x0::Union{AbstractArray,Nothing}=nothing
    )
        x0 = x0===nothing ? b .+ 0.1 : x0
        r = reshape(b - A(x0), :)
        T = typeof(r[1])            # type extraction          
        this = new{T}()
        this.r = r
        this.A = A
        this.b = b
        this.tensor_size = size(b)
        this.x = x0
        # this.r = ComputeResidualVec(this, this.x)
        this.rnew = similar(this.r)
        this.d = this.r
        this.Ad = similar(this.r)
        this.x = x0
        this.itr = 0

        this.Q = Vector{Vector{T}}()
        push!(this.Q, this.r/norm(this.r))

        this.storage_limit = length(this.r) - 1       # Maximal limit, if not, we have a problem. 
        this.reorthogonalize = true
        this.ortho_period = 1
        return this
    end

    function ConjGradModified(A::AbstractMatrix, b::AbstractVector, x0::Union{AbstractArray,Nothing}=nothing)
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
    vec .= this.A(reshape(x, this.tensor_size))
    return reshape(vec, :)
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
    ComputeVec!(this, d, Ad)
    a = dot(r, r)/dot(d, Ad)
    
    if a < 0 
        error("CG got a non-definite matrix")
    end

    this.x += a*d
    this.rnew .= r - a*Ad                    # update rnew 
    rnewNorm = norm(this.rnew)
    
    if this.reorthogonalize 
        orErr = abs(dot(this.Q[1], this.rnew/rnewNorm))
        if this.itr%this.ortho_period == 0
            println("Itr = $(this.itr); re-orthog error $orErr")
            for q in this.Q 
                this.rnew .-= dot(q, this.rnew)*q
            end
            this.d = this.rnew  # Basically restart
        else
            b = dot(this.rnew, this.rnew)/dot(r, r)
            this.d = this.rnew + b*d
        end
        
        # manage the vectors. 
        if length(this.Q) == this.storage_limit
            newQ = popfirst!(this.Q)
            newQ .= this.rnew/rnewNorm
            push!(this.Q, newQ)
        else
            push!(this.Q, this.rnew/rnewNorm)
        end
    end
    
    # @assert abs(dot(rnew + ??*d, Ad)) < 1e-8 "Not conjugate"
    
    this.r = copy(this.rnew)                      # Override
    this.itr += 1 
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

using LinearAlgebra