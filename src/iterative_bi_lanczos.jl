# This is an impelementations of the bi-lanczos algorithms, producing an 
# left and right decomposition of the matrix that is orthogonal, and transform 
# the matrix to a triangular one. 
# Major references are from Professor GreenBaum's book and my own derivations.

mutable struct IterativeBiLanczos

    A::Function                 # the A matrix.
    AH::Function                # The A hermitian matrix 
    T::Dict{Tuple, Number}      # The tridiagonal matrix. 
    V::Dict{Int64, Union{Number, AbstractVector}}   # The right krylov subspace. 
    W::Dict{Int64, Union{Number, AbstractVector}}   # The left krylov subspace. 

    tensor_shape::Tuple
    Av::AbstractVector          # Computed results
    AHw::AbstractVector         

    itr::Int64                  # Iterations Counts 
    """
        The initializations needs: 
            * A linear operator: A in the form of a function, acts on tensor. 
            * 2 initialization vector r1, r2, for the krylov subspace of A and A^H. 
    """
    function IterativeBiLanczos(
        A::Function,
        AH::Function, 
        r1::AbstractArray, 
        r2::AbstractArray
    )
        this = new()
        @assert dot(r1, r2) != 0 "The 2 initialization vector for bi-lanczos cannot be orthogobal to each other. "
        r1 /= norm(r1)
        r2 /= dot(r2, r1)
        this.A = A; this.AH = AH
        Av = this.A(r1); AHw = this.AH(r2)   # try left and right apply the vector to the linear operator. 
        @assert size(Av) == size(AHw) "The shape of left and right krylov ortho tensor should be in the same shape. "
        this.Av = Av; this.AHw = AHw
        # Initialization of T, tridiag. 
        this.T = Dict{Tuple, typeof(Av[1])}()
        this.T[0, 1] = 0             # Gamma zero
        this.T[1, 0] = 0             # beta zero
        this.T[1, 1] = dot(Av, r2)   # alpha one 
        # Initializaiton of V, W ortho subspace. 
        this.V = Dict{Int64, Union{AbstractVector, Number}}(); 
        this.W = Dict{Int64, Union{AbstractVector, Number}}()
        this.V[0] = 0; this.W[0] = 0
        this.V[1] = r1; this.W[1] = r2
        # Misc Parameters. 
        this.tensor_shape = size(Av)
        this.itr = 0
        return this
    end

    function IterativeBiLanczos(A::AbstractMatrix{T}) where {T<:Number}
        AFunc = (x) -> A*x
        AHFunc = (x) -> A'*x
        r1 = rand(T, size(A, 1))
        r2 = rand(T, size(A', 1))
        return IterativeBiLanczos(AFunc, AHFunc, r1, r2)
    end
end

function RightApply(this::IterativeBiLanczos, v::AbstractVector)
    return reshape(this.A(reshape(v, this.tensor_shape)), :)
end

function LeftApply(this::IterativeBiLanczos, v::AbstractVector)
    return reshape(this.AH(reshape(v, this.tensor_shape)), :)
end


function GetTMatrix(this::IterativeBiLanczos, start::Int64 = 1, stop = nothing)
    stop = stop === nothing ? this.itr : stop
    T = this.T
    valuesX = Vector{Int64}()
    valuesY = Vector{Int64}()
    values = Vector{Number}()
    for Idx in start: stop
        β = T[Idx, Idx + 1]
        push!(valuesX, Idx); push!(valuesY, Idx + 1); push!(values, β)
        α = T[Idx, Idx]
        push!(valuesX, Idx); push!(valuesY, Idx); push!(values, α)
        γ = T[Idx + 1, Idx]
        push!(valuesX, Idx + 1); push!(valuesY, Idx); push!(values, γ)
    end
    # Only on element in the matrix I think
    if this.itr == 0
        return T[1, 1]
    end
    push!(valuesX, stop + 1)
    push!(valuesY, stop + 1)
    push!(values, T[stop + 1, stop + 1])
    return sparse(valuesX, valuesY, values)
end

function GetWMatrix(this::IterativeBiLanczos)
    return GetWMatrix(this, 1, this.itr)
end

function GetWMatrix(this::IterativeBiLanczos, start::Int64, stop::Int64)
    return hcat([this.W[Idx] for Idx in start: stop + 1]...)
end

function GetVMatrix(this::IterativeBiLanczos)
    return GetVMatrix(this, 1, this.itr)
end

function GetVMatrix(this::IterativeBiLanczos, start::Int64, stop::Int64)
    return hcat([this.V[Idx] for Idx in start: stop + 1]...)
end


"""
    Returns the matrix, V, T, and W. 
    A = VTW^T. 
"""
function Decomposition(this::IterativeBiLanczos)
    return nothing
end

"""
    Performs one step of the iterative bi-lanczos algorithm and 
    return the v, w matrix. 
"""
function (this::IterativeBiLanczos)()
    this.itr += 1
    j = this.itr

    Av = this.Av; AHw = this.AHw
    V = this.V; W = this.W
    T = this.T
    α = T[j, j]
    ṽ = similar(Av)
    w̃ = similar(AHw)
    @. ṽ =  Av - α*V[j] - T[j - 1, j]*V[j - 1]
    @. w̃ = AHw - conj(α)*W[j] - T[j, j - 1]*W[j - 1]
    # Invariant space might happen here, but due to numerical stability, we don't know 

    γ = norm(ṽ)
    v = ṽ/γ  # v_{j + 1} set up here. 
    β = dot(v, w̃)
    if β == 0
        error("BiLanczos serious break down, both v, w are non-zero but inner product is zero. ")
    end
    w = w̃/conj(β)
    
    this.T[j, j + 1] = β
    this.T[j + 1, j] = γ
    this.W[j + 1] = w
    this.V[j + 1] = v

    # update variables for the next run. 
    this.Av = RightApply(this, V[j + 1])
    this.AHw = LeftApply(this, W[j + 1])
    this.T[j + 1, j + 1] = dot(RightApply(this, v), w)  # Update alpha. 
    
    return v, w
end


"""
    Performs the iterations for j step, and returned the T, W, and V
    matrix generated during the running of the algorithms. 
"""
function (this::IterativeBiLanczos)(j::Int64)
    start = this.itr + 1 
    for _ in 1: j
        this()
    end
    return GetVMatrix(this, start, this.itr), 
        GetWMatrix(this, start, this.itr), 
        GetTMatrix(this, start)
end

# ==============================================================================
# Basic Testing
