include("CG_Convergence_Utilities.jl")

# ==============================================================================
# Exact Arithematic error bound.
# ==============================================================================


function Experiment1(A, b, immitate_exact=true)
    cgm = ConjGradModified(A, b)
    if !immitate_exact 
        TurnOffReorthgonalize(cgm)
    end
    ResidualNorm = GetResidualNorm(cgm)
    ResidualNorms = Vector{Float64}()
    Guesses = Vector{Vector{Float64}}()
    push!(ResidualNorms, ResidualNorm)
    while ResidualNorm > 1e-10
        ResidualNorm = cgm()
        push!(ResidualNorms, ResidualNorm)
        push!(Guesses, cgm.x)
    end
    return ResidualNorms, Guesses
end


N = 512
A = GetUniformPSDMatrix(N)
b = rand(N)
ResidualNorms, Guesses = Experiment1(A, b)
ResidualEnergy = ResRelEnergyNorm(A, b,Guesses)
k = length(ResidualNorms)
ErrorsBound = [TheoreticalErrorBound(A, idx) for idx in 1: k]

fig1 = plot(log10.(ResidualEnergy), label="Relative Energy (exact)", legend=:bottomleft)
plot!(fig1, log10.(ErrorsBound), label="Theoretical Error Bound")
display(fig1)
SaveFigToCurrentScriptDir(fig1, "fig1.png")

# ==============================================================================
# Floating Points Error Bound.
# ==============================================================================

A = GetUniformPSDMatrix(N)
b = rand(N)
ResidualNorms, Guesses = Experiment1(A, b, false)
ResidualEnergy = ResRelEnergyNorm(A, b,Guesses)
k = length(ResidualNorms)
ErrorsBound = [TheoreticalErrorBound(A, idx) for idx in 1: k]

fig2 = plot(log10.(ResidualEnergy), label="Relative Energy (floats)", legend=:bottomleft)
plot!(fig2, log10.(ErrorsBound), label="Theoretical Error Bound(exact)")
display(fig2)
SaveFigToCurrentScriptDir(fig2, "fig2.png")
