using LinearAlgebra, SparseArrays
using Logging
using Plots


# Suitable for any linear transformation on any type of multi-dimensional arrays. 
# Implementation details and advantages: 
# * has the option to keep all previous residual vectors for orthogonalizing the newest 
#   residual vectors obtained. 
# * The re-orthogonalization process is computed in parallel because it uses a matrix of zero that 
#   resizes to double the number of columns each time.  

mutable struct ConjGrad
    A::Function              # Linear opeartor
    b::AbstractArray         # RHS vector
    tensor_size::Tuple       # The size of the tensor the linear operator is acting on. 

    x::AbstractVector        # current guess, started with initial guess. 
    r::AbstractVector        # previous computed residual
    rnew::AbstractVector     # Current residual
    d::AbstractVector        # Conjugate Direction
    
    itr::UInt64              # Iteration count. 
    Q::Union{AbstractMatrix, Nothing}        # Re-Orthogonalization basis.
    Q_size::UInt64           # Number of orthogonalization vectors. 
    over_write::UInt64       # position for overwrite once storage is limit is reached.

    storage_limit::UInt64    # storage limit for the Q vector. 
    reorthogonalize::Bool    # Whether to perform reorthogonalization.
    function ConjGrad(
        A::Function, 
        b::AbstractArray, 
        x0::Union{AbstractArray,Nothing}=nothing
    )
        this = new()
        x0 = x0===nothing ? b .+ 0.1 : x0
        this.A = A
        this.b = b
        this.tensor_size = size(b)
        this.x = x0
        this.r = ComputeResidualVec(this, this.x)
        this.rnew = similar(this.r)
        this.d = this.r
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

    function ConjGrad(A::AbstractMatrix, b::AbstractVector, x0::Union{AbstractArray,Nothing}=nothing)
        return ConjGrad((x)-> A*x, b, x0)
    end

end


"""
    Turn on the reorthogonalizations using the residual vectors, 
    and add current residual to the list of residuals. 
"""
function TurnOnReorthgonalize(this::ConjGrad)
    this.reorthogonalize = true
    this.Q = zeros(typeof(this.r[1]), length(this.r), 4)
    this.Q[:, 1] = this.r/norm(this.r)
    this.Q_size = 1
    this.over_write = 0
    return 
end

"""
    Turn off the reorthogonalization on the residual vectors. And then 
    clear all the stored residual vectors. 
"""
function TurnOffReorthgonalize(this::ConjGrad)
    this.reorthogonalize = false
    this.Q = nothing
    return
end


"""
    Change the storage limit for the number of vectors used for 
    re-orthogonalization.
"""
function ChangeStorageLimit(this::ConjGrad, storage_limit)
    this.storage_limit = min(length(this.r) - 1, storage_limit)
    return
end

"""
    Apply the linear transformation on the tensor and return the 
    residual vector after transformation. 
"""
function ComputeResidualVec(this::ConjGrad, x::AbstractArray)
    resVec = this.b - this.A(reshape(x, this.tensor_size))
    return reshape(resVec, length(resVec))
end


"""
    Compute the vector using the linear transformation. 
"""
function ComputeVec(this::ConjGrad, x::AbstractArray)
    vec = this.A(reshape(x, this.tensor_size))
    return reshape(vec, length(vec))
end


"""
    Performs One step of conjugate Gradient. 
"""
function (this::ConjGrad)()
    r = this.r
    if norm(r) == 0
        return 0 # The problem is solved already. 
    end
    d = this.d
    Ad = ComputeVec(this, d)
    a = dot(r, r)/dot(d, Ad)
    if a < 0 
        error("CG got a non-definite matrix")
    end

    this.x += a*d
    this.rnew = r - a*Ad                    # update rnew 
    
    if this.reorthogonalize
        this.rnew -= this.Q*this.Q'*this.rnew

        if this.Q_size == this.storage_limit  # starts overwrite
            this.Q[:, this.over_write + 1] = this.rnew/norm(this.rnew)
            this.over_write = (this.over_write + 1)%this.storage_limit

        elseif this.Q_size == size(this.Q, 2) # Resize
            newQ = zeros(
                typeof(this.r[1]), 
                size(this.Q, 1), 
                max(2*size(this.Q, 2), this.storage_limit)
            )
            newQ[:, 1:this.Q_size] = this.Q
            newQ[:, this.Q_size + 1] = this.rnew/norm(this.rnew)
            this.Q = newQ
            this.Q_size += 1
        else                                  # add one more.
            this.Q[:, this.Q_size + 1] = this.rnew/norm(this.rnew)
            this.Q_size += 1
        end
    end
    
    b = dot(this.rnew, this.rnew)/dot(r, r)
    # @assert abs(dot(rnew + β*d, Ad)) < 1e-8 "Not conjugate"
    this.d = this.rnew + b*d
    this.r = this.rnew                      # Override
    this.itr += 1 
    return convert(Float64, norm(this.rnew))
end


### Experiments. 

function GetErrorEnergyNorm(A, b, xs)
    xStar = A\b
    return xs .|> x -> dot(x - xStar, A, x-xStar)/dot(xs[1] - xStar, A, xs[1] - xStar)
end

"""
    Bad eigen value distribution, they are distributed from 0.001 to 1 like a 
    geometric series. 
     
    * A lot of small eigenvalues are clustered close to zero, few larger ones are 
    far away from the other Eigenvalues. 

    Parameters:
        rho: Number: 
            An number between 0 and 1 that paramaterize the distribution of the 
            eigenvalues for the PSD matrix. 
        N=20: 
            The size of the PSD matrix.
    
    returns: 
        a diagonal matrix. 
"""
function GetNastyPSDMatrix(rho::Number, N=20, inverted=false)
    @assert rho <= 1 && rho >= 0
    EigenValues = zeros(N)
    EigenMin, EigenMax = 0.001, 1    # Min Max Eigenvalues. 
    EigenValues[1] = EigenMin
    for IdexI in 2:N
        EigenValues[IdexI] = EigenMin + 
            ((IdexI - 1)/(N - 1))*(EigenMax - EigenMin)*rho^(N - IdexI)  # formulas
    end
    if inverted
        return Diagonal(1 + EigenMin - EigenValues)
    end
    return Diagonal(EigenValues)
end


N = 256
A = GetNastyPSDMatrix(0.9, N)
B = sprand(Float64, N, N, 2/N)
A = A + B'*B
b = rand(N)
cg1 = ConjGrad(A, b)
cg2 = ConjGrad(A, b)
cg3 = ConjGrad(A, b)
cg1Soln = Vector{typeof(b)}()
cg2Soln = Vector{typeof(b)}()
cg3Soln = Vector{typeof(b)}()
push!(cg1Soln, cg1.x)
push!(cg2Soln, cg2.x)
push!(cg3Soln, cg3.x)
TurnOffReorthgonalize(cg2)
ChangeStorageLimit(cg3, 188)
for _ in 1:3*N
    cg1(); cg2(); cg3()
    push!(cg1Soln, cg1.x); push!(cg2Soln, cg2.x); push!(cg3Soln, cg3.x)
end

objectiveVals1 = GetErrorEnergyNorm(A, b, cg1Soln)
objectiveVals2 = GetErrorEnergyNorm(A, b, cg2Soln)
objectiveVals3 = GetErrorEnergyNorm(A, b, cg3Soln)
fig = plot(log10.(objectiveVals1), label="With Re-orthonalization")
plot!(fig, log10.(objectiveVals2), label="without Re-orthogonalization")
plot!(fig, log10.(objectiveVals3), label="partial Re-orthogonalization")
