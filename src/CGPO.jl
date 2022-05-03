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
    AP::Vector{Vector{T}}   # Past AP vector. 
    R::Vector{Vector{T}}    # past residual vectors. 
    storage_limit           # How many P vectors to store. 
    orthon_on::Bool   
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
        this.rnew = this.r
        this.p = this.r
        this.Ap = similar(this.r)
        this.x = x0
        this.itr = 0
        this.P = Vector{typeof(r)}()
        this.AP = Vector{typeof(r)}()
        this.R = Vector{typeof(r)}()
        r̂ = this.r/norm(this.r)
        push!(this.P, r̂)
        push!(this.R, r̂)

        # Default settings
        this.storage_limit = length(this.r) - 1     # Maximal limit, if not, we have a problem. 
        this.orthon_on = true

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
    Evalute one step of the CGPO, perform conjugation processs according to 
    settings. 
"""
function (this::CGPO)()
    r = this.r
    if norm(r) == 0
        return 0 
    end

    p = this.p
    Ap = this.Ap
    Ap = ComputeVec(this, p)   # <-- introduces error
    a = dot(r, r)/dot(p, Ap)   # Error here

    if a < 0 
        warn("reset step size a. ")
        a = 0
    end

    this.x += a*p
    this.rnew = r - a*Ap      # <--- Here. 
    rnew_dotted = dot(this.rnew, this.rnew)
    rnewNorm = sqrt(rnew_dotted)
    b = rnew_dotted/dot(r, r)  # <--- Here too. 
    
    this.p = this.rnew + b*p   
    
    # partial orthogonalizations only errors on r, p is fixed here. 
    
    δp = zeros(size(this.r))
    δr = zeros(size(this.r))
    Ar = ComputeVec(this, this.rnew)
    if this.orthon_on
        
        for p̄ in this.P[1: end - 1]
            Ap̄ = ComputeVec(this, p̄)
            δp -= (dot(p̄, Ar)/dot(p̄,Ap̄))*p̄
        end
        for r̄ in this.R[1: end]
            δr += dot(r̄, this.rnew)*r̄
        end
        δp -= (dot(this.r, this.rnew)/dot(this.r, this.r))*this.P[end]
        this.rnew -= δr
        this.p += δp
        
    end

    this.r = copy(this.rnew)
    UpdatePR!(this, this.p, this.rnew/norm(this.rnew))
    this.itr += 1

return convert(Float64, rnewNorm) end

function GetPMatrix(this::CGPO) return hcat(this.P...) end

function ResNorm(this::CGPO) return convert(Float64, norm(this.rnew)) end

function TurnOffReorthgonalize!(this::CGPO)
    this.orthon_on = false
    empty!(this.P)
    empty!(this.R)
return end

"""
    Keep currrent x, and reset everything that is here. 
"""
function Reset!(this::CGPO)
    this.r = this.b - ComputeVec(this, this.x)
    this.p = this.r
    empty!(this.P)
return end

function StorageLimit!(this::CGPO, limit::Int)
    if limit < 1
        return # simply ignore. 
    end
    this.storage_limit = limit
return end

function UpdatePR!(this::CGPO, new_p, new_r)
    while length(this.P) > this.storage_limit
        popfirst!(this.P)
        popfirst!(this.R)
    end
    push!(this.P, new_p)
    push!(this.R, new_r)
return end



# using LinearAlgebra, Plots

# function BasicRun()
#     N = 16; ϵ = 1e-10
#     d = LinRange(0, 1 - ϵ, N)
#     A = Diagonal(d.^2 .+ ϵ)
#     b = ones(N)
#     ẋ = A\b
#     cg1 = CGPO(A, b)
#     cg1.storage_limit = N/2
#     cg3 = CGPO(A, b)
#     cg3.orthon_on = true
#     e0 = ẋ - cg1.x
#     P1 = Vector{Vector{Float64}}()
#     push!(P1, cg1.p)
#     for II in 1:N^2
#         cg1()
#         push!(P1, cg1.p)
#         e = ẋ - cg1.x
#         if sqrt(dot(e,A*e))/sqrt(dot(e0, A*e0)) < 1e-1
#             println("cg1 Iter: $(II)")
#             break
#         else
#             # println(norm(ẋ - cg1.x, Inf))
#         end
#     end
#     P1 = hcat(P1...)
#     D1 = P1'*A*P1
#     display(heatmap(D1.|>abs.|>log2))


#     P3 = Vector{Vector{Float64}}()
#     push!(P3, cg3.p)
#     for II in 1:N^2
#         cg3()
#         e = ẋ - cg3.x
#         push!(P3, cg3.p)
#         if sqrt(dot(e, A*e))/sqrt(dot(e0, A*e0))  < 1e-1
#             println("cg3 Iter: $(II)")
#             break
#         else
#             # println(norm(ẋ - cg3.x, Inf))
#         end
#     end
#     P3 = hcat(P3...)
#     D3 = P3'*A*P3
#     display(heatmap(D3.|>abs.|>log2))

    
# return end

# BasicRun()