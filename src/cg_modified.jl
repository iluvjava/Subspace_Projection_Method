# Suitable for any linear transformation on any type of multi-dimensional arrays. 
# Implementation details and advantages: 
# * has the option to keep all previous residual vectors for orthogonalizing the newest 
#   residual vectors obtained. 
# 
# * The re-orthogonalization process is computed in parallel because it uses a matrix of zero that 
#   resizes to double the number of columns each time. 
# 
# * (TO ADD) Always store the Tridiaognalizations of the Linear operator while running the conjugate gradient. 

mutable struct ConjGradModified{T <: Number}
    A::Function              # Linear opeartor
    b::AbstractArray{T}      # RHS vector
    tensor_size::Tuple       # The size of the tensor the linear operator is acting on. 

    x::Vector{T}            # current guess, started with initial guess. 
    r::Vector{T}            # previous computed residual
    rnew::Vector{T}         # Current residual
    d::Vector{T}            # Conjugate Direction
    Ad::Vector{T}           # for reducing garbage collector time. 
    
    itr::UInt64              # Iteration count. 
    Q::Union{AbstractMatrix{T}, Nothing}        # Re-Orthogonalization basis.
    Q_size::UInt64           # Number of orthogonalization vectors. 
    over_write::UInt64       # position for overwrite once storage is limit is reached.

    storage_limit::UInt64    # storage limit for the Q vector. 
    reorthogonalize::Bool    # Whether to perform reorthogonalization.
    

    function ConjGradModified(
        A::Function, 
        b::AbstractArray, 
        x0::Union{AbstractArray,Nothing}=nothing
    )
        x0 = x0===nothing ? b .+ 0.1 : x0
        r = reshape(b - A(x0), :)
        this = new{typeof(r[1])}()
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

        this.Q = zeros(typeof(this.r[1]), length(this.r), 4)
        this.Q[:, 1] = this.r/norm(this.r)
        this.Q_size = 1
        this.over_write = 0

        this.storage_limit = length(this.r) - 1       # Maximal limit, if not, we have a problem. 
        this.reorthogonalize = true
        return this
    end

    function ConjGradModified(A::AbstractMatrix, b::AbstractVector, x0::Union{AbstractArray,Nothing}=nothing)
        return ConjGradModified((x)-> A*x, b, x0)
    end

end


"""
    Turn on the reorthogonalizations using the residual vectors, 
    and add current residual to the list of residuals. 
"""
function TurnOnReorthgonalize(this::ConjGradModified)
    this.reorthogonalize = true
    this.Q = zeros(typeof(this.r[1]), length(this.r), storage_limit)
    this.Q[:, 1] = this.r/norm(this.r)
    this.Q_size = 1
    this.over_write = 0
    return 
end


"""
    Turn off the reorthogonalization on the residual vectors. And then 
    clear all the stored residual vectors. 
"""
function TurnOffReorthgonalize(this::ConjGradModified)
    this.reorthogonalize = false
    this.Q = nothing
    return
end


"""
    Change the storage limit for the number of vectors used for 
    re-orthogonalization.
"""
function ChangeStorageLimit(this::ConjGradModified, storage_limit)
    this.storage_limit = min(length(this.r) - 1, storage_limit)
    return
end



"""
    Set the number of vectors in the Q matrix by force. This will clear 
    everything that is already in Q. 
"""
function SetStorageLimit(this::ConjGradModified, storage_limit)
    this.Q = zeros(typeof(this.r[1]), length(this.r), min(length(this.r) - 1, storage_limit))
    this.Q[:, 1] = this.r/norm(this.r)
    this.storage_limit = storage_limit
    this.Q_size = 1
    this.over_write = 0
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
        this.rnew .-= this.Q*(this.Q'*this.rnew)
        
        if this.Q_size == this.storage_limit  # starts overwrite
            this.Q[:, this.over_write + 1] = this.rnew/rnewNorm
            this.over_write = (this.over_write + 1)%this.storage_limit
        elseif this.Q_size == size(this.Q, 2) # Resize
            newQ = zeros(
                typeof(this.r[1]), 
                size(this.Q, 1), 
                min(2*size(this.Q, 2), this.storage_limit)
            )
            newQ[:, 1:this.Q_size] = this.Q
            newQ[:, this.Q_size + 1] = this.rnew/rnewNorm
            this.Q = newQ
            this.Q_size += 1
        else                                  # add one more.
            this.Q[:, this.Q_size + 1] = this.rnew/rnewNorm
            this.Q_size += 1
        end
    end
    
    b = dot(this.rnew, this.rnew)/dot(r, r)
    # @assert abs(dot(rnew + β*d, Ad)) < 1e-8 "Not conjugate"
    this.d = this.rnew + b*d
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

