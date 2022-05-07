using LinearAlgebra, SparseArrays
using Plots

mkpath("$(@__DIR__)/plots")

"""
    Given the spetrum of a matrix (a vector), this function returns a Diagonal matrix 
    whose diagonals are the same as the given vector, and then it returns 
    the matrix, along with a perturbed matrix. 
"""
function TinyIntervalTestMatrices(
    v::Vector{Float64}; 
    scale=55, 
    delta=eps(Float64), 
    m = 10
)
    A = Diagonal(v)
    u = Vector{Number}()
    for l in v
        w = LinRange(l - scale*delta, l + scale*delta, m)|>collect
        append!(u, w...) 
    end
return A, Diagonal(u) end


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
function EigDistribution(
    N=64;
    rho::Number=0.9,
    eigen_min=1e-4, 
    eigen_max=1, 
    inverted=false
)
    @assert rho <= 1 && rho >= 0
    EigenValues = zeros(N)
    EigenValues[1] = eigen_min
    for IdexI in 2:N
        EigenValues[IdexI] = eigen_min + 
            ((IdexI - 1)/(N - 1))*(eigen_max - eigen_min)*rho^(N - IdexI)  # formulas
    end
    if inverted
        return Diagonal(1 + eigen_min - EigenValues)
    end
    return EigenValues
end

"""
    Perform CG and collect the relative error under energy norm
"""
function PerformCGFor(
    A::AbstractMatrix, 
    b::AbstractVecOrMat;
    epsilon=1e-2,       # tolerance of relative energy norm of the error 
    exact::Bool=true,   # Whether to use full orthogonalizations
    partial_ortho=nothing,  # if partial, how many previous iterations? 
)
    cg = CGPO(A, b)
    if !exact
        if partial_ortho === nothing
            cg |> TurnOffReorthgonalize!
        else
            StorageLimit!(cg, partial_ortho)
        end
    end

    ẋ = A\b  # direct soooooolve! 
    ė = ẋ - cg.x
    ėAė = dot(ė, A*ė)
    E = Vector{Float64}()
    push!(E, 1)
    RelErr = 1
    while RelErr > epsilon
        cg()
        e = ẋ - cg.x
        RelErr = (dot(e, A*e)/ėAė)|>sqrt
        push!(E, RelErr)
    end
    return E
end