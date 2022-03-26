### ----------------------------------------------------------------------------
### Conjugate Gradient with Re-Orthogonalizations
### ----------------------------------------------------------------------------
### Algorithm Descriptions: 
###     To keep the search directions conjugate, we use Gramschitz Orthogonalizations
###     on all previous conjugate vectors. This struct supports the options of 
###     choosing a fixed number of vectors to store by. 



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
    R::Vector{Vector{T}}    # past residual vectors. 
    storage_limit           # How many P vectors to store. 
    orthogonalization_mode::Int64   
                            # An options for how to orthogonalized. 
                            # 1: Partial orthogonalize according to storage 
                            # limit

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
        this.R = Vector{typeof(r)}()
        r̂ = this.r/norm(this.r)
        push!(this.P, r̂)
        push!(this.R, r̂)

        # Default settings
        this.storage_limit = length(this.r) - 1       # Maximal limit, if not, we have a problem. 
        this.orthogonalization_mode = 1

    return this end

    function CGPO(
        A::AbstractMatrix, 
        b::AbstractArray
    )
    return CGPO((x) -> A*x, b) end
end


"""
    Pre-allocates a vector and get the values to the given 
    pre-allocated vector. 
"""
function ComputeVec(this::CGPO, x::AbstractArray)
    vec = this.A(reshape(x, this.tensor_size))
return reshape(vec, :) end


"""
    Evalute One step of the CGPO, perform conjugation processs according to 
    settings. 
"""
function (this::CGPO)()
    r = this.r
    if norm(r) == 0
        return 0 
    end

    p = this.p
    Ap = this.Ap
    Ap = ComputeVec(this, p)
    a = dot(r, r)/dot(p, Ap)

    if a < 0 
        warn("reset step size a. ")
        a = 0
    end

    this.x += a*p
    this.rnew .= r - a*Ap
    rnew_dotted = dot(this.rnew, this.rnew)
    rnewNorm = sqrt(rnew_dotted)
    b = rnew_dotted/dot(r, r)
    
    this.p = this.rnew + b*p
    
    # partial orthogonalizations. 
    Ar = ComputeVec(this, this.rnew)
    δp = zeros(size(this.r))
    δr = zeros(size(this.r))
    if this.orthogonalization_mode == 1
        for p̄ in this.P[
                convert(Int64, max(length(this.P) - this.storage_limit, 1)): end - 1
            ]
            Ap̄ = ComputeVec(this, p̄)
            δp -= (dot(p̄, Ar)/dot(p̄,Ap̄))*p̄
        end
        this.p += δp
        for r̄ in this.R[
            convert(Int64, max(length(this.R) - this.storage_limit, 1)): end - 1
        ]
            δr += dot(r̄, this.rnew)*r̄
        end
        this.rnew -= δr
    end

    this.r = copy(this.rnew)
    push!(this.P, this.p)
    push!(this.R, this.rnew/norm(this.rnew))
    this.itr += 1
    AutoTrimStorage!(this)

return convert(Float64, rnewNorm) end

function GetPMatrix(this::CGPO) return hcat(this.P...) end

"""
    Trim the storage for all the conjugate vectors. 
"""
function AutoTrimStorage!(this::CGPO)
    # while length(this.P) > this.storage_limit
    #     popfirst!(this.P)
    # end
end

function Reset!(this::CGPO)
    this.r = this.b - ComputeVec(this, this.x)
    this.p = this.r
    empty!(this.P)
return end


# Basic Tests ------------------------------------------------------------------

using LinearAlgebra, Plots

function BasicRun()
    N = 1024; ϵ = 1e-10
    d = LinRange(0, 1 - ϵ, N)
    A = Diagonal(d .+ ϵ)
    b = ones(N)
    ẋ = A\b
    cg1 = CGPO(A, b)
    cg1.storage_limit = N
    cg3 = CGPO(A, b)
    cg3.orthogonalization_mode = 0
    e0 = ẋ - cg1.x
    for II in 1:N^2
        cg1()
        e = ẋ - cg1.x
        if sqrt(dot(e,A*e))/sqrt(dot(e0, A*e0)) < 1e-10
            println("cg1 Iter: $(II)")
            break
        else
            # println(norm(ẋ - cg1.x, Inf))
        end
    end
    P1 = GetPMatrix(cg1)
    D1 = P1'*A*P1
    display(heatmap(D1.|>abs.|>log2))
    for II in 1:N^2
        cg3()
        e = ẋ - cg3.x
        if sqrt(dot(e, A*e))/sqrt(dot(e0, A*e0))  < 1e-10
            println("cg3 Iter: $(II)")
            break
        else
            # println(norm(ẋ - cg3.x, Inf))
        end
    end
    P3 = GetPMatrix(cg3)
    D3 = P3'*A*P3
    display(heatmap(D3.|>abs.|>log2))

    
return end

BasicRun()
