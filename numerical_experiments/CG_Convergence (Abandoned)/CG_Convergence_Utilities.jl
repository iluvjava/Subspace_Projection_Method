using LinearAlgebra, SparseArrays, Logging, Plots, ProgressMeter
include("../../src/cg_modified.jl")

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
function GetNastyPSDMatrix(
    N=20, 
    rho::Number=0.9, 
    eigen_min=1e-3, 
    eigen_max=1, 
    inverted=false
)
    @assert rho <= 1 && rho >= 0
    EigenValues = zeros(N)
    eigen_min, eigen_max = 0.001, 1    # Min Max Eigenvalues. 
    EigenValues[1] = eigen_min
    for IdexI in 2:N
        EigenValues[IdexI] = eigen_min + 
            ((IdexI - 1)/(N - 1))*(eigen_max - eigen_min)*rho^(N - IdexI)  # formulas
    end
    if inverted
        return Diagonal(1 + eigen_min - EigenValues)
    end
    return Diagonal(EigenValues)
end


"""
    Get a diagonal PSD matrix whose eigenvalues are squared of a standard
    normal distributions. 
"""
function GetNormalPSDMatrix(N=20)
    v = randn(N).^2
    return Diagonal(v) 
end

"""

"""
function GetUniformPSDMatrix(N)
    v = rand(N)
    return Diagonal(v)
end



"""
    Given the opeartr, RHS vector, and the list of guesses, compute the 
    relative error in energy norm induced by linear operator A. 
"""
function ResRelEnergyNorm(A, b, xs)
    x̄ = A\b
    x0 = xs[1]
    return xs .|> x -> dot(x - x̄, A, x - x̄)/dot(x0 - x̄, A, x0 - x̄)
end



"""
    Compute the Theoretical Error Bound for a given Linear Opeartor and the 
    iterations count that you are looking at. 
"""
function TheoreticalErrorBound(A::AbstractMatrix, k)
    κ = cond(A)
    return 2*((sqrt(κ) - 1)/(sqrt(κ) + 1))^k
end


"""
    Given an instance of a plot, and the file name, it 
    stores it to a folder named: 'plots' under the 
    directory of the running script. 
"""
function SaveFigToCurrentScriptDir(fig::Plots.Plot, file_name::String)
    scriptDir = @__DIR__
    directory = scriptDir*"/plots"
    path = mkpath(directory)
    savefig(fig, directory*"/"*file_name)
    return path
end

