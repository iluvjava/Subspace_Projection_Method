### ----------------------------------------------------------------------------
### Conjugate Gradient with Re-Orthogonalizations
### ----------------------------------------------------------------------------
### Algorithm Descriptions: 
###     To keep the search directions conjugate, we use Gramschitz Orthogonalizations
###     on all previous conjugate vectors. This struct supports the options of 
###     choosing a fixed number of vectors to store by. 
### 
###


mutable struct CGPO{T <: Number}
    A::Function             # Linear opeartor
    b::AbstractArray{T}     # RHS vector
    tensor_size::Tuple      # The size of the tensor the linear operator is acting on. 

    x::Vector{T}            # current guess, started with initial guess. 
    r::Vector{T}            # previous computed residual
    rnew::Vector{T}         # Current residual
    p::Vector{T}            # Conjugate Direction
    Ap::Vector{T}           # for reducing garbage collector time. 
    
    itr::UInt64             # Iteration count. 
    P::Vector{Vector{T}}    # Past Conjugate Vectors.
    storage_limit           # How many P vectors to store. 
    orthogonalization_mode::Int64   
                            # An options for how to orthogonalized. 
                            # 0: fullly orthogonalize on all P, ignore storage
                            # limit
                            # 1: Partial orthogonalize according to storage 
                            # limit
                            # 2: Smart orthogonalize using storage limit

    function CGPO(
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
        this.p = this.r
        this.Ap = similar(this.r)
        this.x = x0
        this.itr = 0

        this.P = Vector{typeof(r)}()
        push!(this.P, this.r/norm(this.r))

        # Default settings
        this.storage_limit = length(this.r) - 1       # Maximal limit, if not, we have a problem. 
        this.reorthogonalize = 0

    return this end
end

"""
    Pre-allocates a vector and get the values to the given 
    pre-allocated vector. 
"""
function ComputeVec!(this::ConjGradModified, x::AbstractArray, vec::AbstractArray)
    vec .= this.A(reshape(x, this.tensor_size))
return reshape(vec, :) end

"""
    Evalute One step of the CGPO, perform conjugation processs according to 
    settings. 
"""
function (this::ConjGradModified)()



return end



# Basic Tests ------------------------------------------------------------------

using LinearAlgebra