include("../demo_utilities.jl")
include("../../src/cg_modified.jl")
@info "Demo CG Modified Script running at $(@__DIR__)"
println("Chaging pwd to current directory of the script")
cd(@__DIR__)
@info "New pwd: $(pwd())"

function GetObjectiveVals(A, b, xs)
    return xs .|> x -> dot(x, A, x) - 2*dot(b, x)
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

N = 512
A = GetNastyPSDMatrix(0.9, N)
A = A^4
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
ChangeStorageLimit(cg3, 32)
for _ in 1:2*N
    cg1(); cg2(); cg3()
    push!(cg1Soln, cg1.x); push!(cg2Soln, cg2.x); push!(cg3Soln, cg3.x)
end

objectiveVals1 = GetObjectiveVals(A, b, cg1Soln)
objectiveVals2 = GetObjectiveVals(A, b, cg2Soln)
objectiveVals3 = GetObjectiveVals(A, b, cg3Soln)
fig = plot(objectiveVals1, label="With Re-orthonalization", title="Convergence of the Objective")
plot!(fig, objectiveVals2, label="without Re-orthogonalization")
plot!(fig, objectiveVals3, label="partial Re-orthogonalization")
savefig(fig, "objectivevals_convergence.png")




cd(".")