# This is reference to some of the ideas in Applied Numerical Linear Algebra, 
# We are investigating a Lanczos the self-correct the lost of orthogonality 
# using ritz vectors projections. 
# Here is what we need to do: 
#   * Testing for the loss of orthgonality during the runtime of the Lanczos algorithm
#   * Once it's detected, start a Eigensolve on SymTridiagonal and then look for well converged Ritz Vectors. 
#   * Store all of those Ritz Vectors as a matrix for reorthogonalizations. 


mutable struct IterativeLanczosModified{T<:Number}
    A::Function                         # Linear Operator on Tensor as a function. 
    T::Dict{Tuple{Int64, Int64}, T}     # Mapping coordinates [i, j] to elements in the Tridiaognal Matrix. 
    k::Int64                            # when k=1, the algorithm haven't started yet. 
    Aq::Vector{T}                       # For temp storage to reduce GC time. 
    Q::Dict{Int64, Vector{T}}           # A Map for storing all the Lanczos Vectors. 
    Q_blocks::Vector{Matrix{T}}         # Blocks of Qs from each restarts
    T_blocks::Vector{Dict}              # Blocks of Ts from each restarts

    tensor_shape::Tuple{Int64}          # Shape of the input for the linear operator. 
    V::Vector{Union{Matrix{T}, Vector{T}}}
                                        # Eigen system of T. 
    Y::Vector{Vector{T}}                # Well converged ritz vectors from last restart. 
    ritz_values::Vector{T}              # The list of ritz values. 
    
    function IterativeLanczosModified(A::Function, q1)
        q1 /= norm(q1)
        Aq = A(q1)
        α = dot(q1, reshape(Aq, :))
        D = typeof(Aq[1])        # Generic Type info
        this = new{D}()          # Referenc to instance created here.
        this.tensor_shape = size(Aq)
        this.Aq = reshape(Aq, :)        # flatten it. 
        this.A = A
        this.Q = Dict{Int64, Vector{D}}()
        this.T = Dict{Tuple{Int64}, D}()
        this.T[1, 1] = α
        this.T[0, 1] = this.T[1, 0] = 0
        this.T_blocks = Vector{AbstractArray{D}}()
        this.Q[1] = q1
        this.Q[0] = zeros(D, length(this.Aq))
        this.Q_blocks = Vector{Matrix{D}}()
        this.k = 1
        this.V = Vector{Union{Matrix{D}, Vector{D}}}()
        this.Y = Vector{Vector{D}}()
        this.ritz_values = Vector{D}()
        return this
    end

    function IterativeLanczosModified(A::AbstractMatrix, q1)
        IterativeLanczosModified((x) -> A*x, q1)
    end

end

# Override the () operator. 
"""
    It performs one iteration of the Lanczos Orthogonalization. 
"""
function (this::IterativeLanczosModified)()
    Aq = this.Aq
    k = this.k
    Q = this.Q
    T = this.T

    q̃ = Aq - T[k - 1, k]*Q[k - 1]
    q̃ -= T[k, k]*Q[k]

    for y in this.Y
        q̃ .-= dot(y, q̃)*y
    end

    β = norm(q̃)
    q = q̃/β

    this.Aq = this.A(q)       
    α = dot(q, this.Aq)
    T[k + 1, k + 1] = α
    T[k, k + 1] = T[k + 1, k] = β
    Q[k + 1] = q
    this.k += 1
    return q
end


"""
    Restart and figure out the Ritz Vectors from the SymTridiaognal Matrix, 
    get the converged ones and put them into the V vectors of matrices. 
"""
function Restart(this::IterativeLanczosModified)
    if this.k == 1                      #  TODO: Investigate Edge Case. 
        return 
    end

    T = GetPreviousT(this, true)        # T is partial
    Q = GetPreviousQ(this)              # Q is full 
    D = typeof(this.Aq[1])
    k = this.k

    this.V = Vector{Union{Matrix{D}, Vector{D}}}()
    this.Y = Vector{Vector{D}}()
    this.ritz_values = Vector{D}()
    Θ, V = eigen(T)
    
    converged = 0
    for (θ, j) in zip(Θ, 1:length(Θ))
        v = V[:, j]
        error = abs(this.T[k, k - 1]*v[k - 1])
        if error < sqrt(eps(Float32))*norm(T)    # store the ritz values and the ritz vector. 
            y = Q[:, 1:end - 1]*v
            ŷ = y/norm(y)
            projVal = 0
            for y in this.Y
                projVal += dot(y, ŷ)
            end
            # if projVal > 1e-10
            #     continue                           # repeated eigenvectors, ignore
            # end
            push!(this.Y, ŷ)
            push!(this.ritz_values, θ)
            converged += 1
        end
    end
    # if converged == 0
    #     return 0
    # end
    
    # # Store Q, reset T, and Q.  
    
    # q1 = Q[:, end]
    # q0 = Q[:, end - 1]
    # α = this.T[k, k]
    # β = this.T[k - 1, k]
    # push!(this.Q_blocks, Q[:, 1:end - 1])
    # push!(this.T_blocks, this.T)
    # this.T = Dict{Tuple{Int64}, D}()
    # this.T[1, 1] = α
    # this.T[0, 1] = this.T[1, 0] = β
    
    # this.Q = Dict{Int64, Vector{D}}()
    # this.Q[1] = q1               # don't need to change Aq
    # this.Q[0] = q0
    # this.k = 1
    return converged
end


"""
    Get the orthogonality error between the newest lanczos vectors 
    and the most recent lanczos vector after the last restart. 
"""
function GetOrthogonalError(this::IterativeLanczosModified)
    return abs(dot(this.Q[1], this.Q[this.k]))
end


"""
    Get the Q matrix generated from the Iterative Lanczos Process. 
"""
function GetQMatrix(this::IterativeLanczosModified)
    return hcat([this.Q_blocks..., GetPreviousQ(this)]...)
end


"""
    Get the Q Matrix since the last restart of the Lanczos Algorithm. 
"""
function GetPreviousQ(this::IterativeLanczosModified)
    return hcat([this.Q[idx] for idx in 1: this.k]...)
end



"""
    Get the Symmetric Tridiagonal matrix from the Iterative Lanczos Process. 
"""
function GetTMatrix(this::IterativeLanczosModified)
    αs = Vector{Number}()
    βs = Vector{Number}()
    for (T, j) in zip([this.T_blocks..., this.T], 1:length(this.T_blocks) + 1)
        blockSize = maximum(tuple[1] for tuple in keys(T))
        α = [T[idx, idx] for idx in 1:blockSize]
        middle = j > 1 && j <= length(this.T_blocks) + 1
        middleStart = middle ? 0 : 1
        β = [T[idx + 1, idx] for idx in middleStart: blockSize - 1]
        append!(αs, α)
        append!(βs, β)
    end
    return SymTridiagonal(αs, βs)
end


"""
    Get the T matrix since last restart of Lanczos Algorithm. 
"""
function GetPreviousT(this::IterativeLanczosModified, ignore_last=false)
    endAt = ignore_last ? this.k - 1 : this.k
    α = [this.T[idx, idx] for idx in 1: endAt]
    β = [this.T[idx + 1, idx] for idx in 1: endAt]
    return SymTridiagonal(α, β)
end



# Basic Testing here -----------------------------------------------------------
using LinearAlgebra, Plots
n = 128
eigenValues = collect(LinRange(1e-3, 1, n)).^4
A = Diagonal(eigenValues); 
b = rand(n)
ilm = IterativeLanczosModified(A, b)
converged = 0
for itr in 1: n - 1
    ilm()
    if itr % 1 == 0
    # if GetOrthogonalError(ilm) > 1e-14
        converged = Restart(ilm)
        println("Restarts at itr=$itr, converged = $converged")
        # if converged >= n
        #     break
        # end
    end
end

Q = GetQMatrix(ilm)
T = GetTMatrix(ilm)
println()
heatmap(Q'*Q .|> abs .|> log2)

# LANSO, 
