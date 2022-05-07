include("./utilities.jl")
include("../../src/CGPO.jl")

N = 64
EIGENS = EigDistribution(N, rho=0.8, eigen_min=1e-5)
A, Ã = TinyIntervalTestMatrices(EIGENS, scale=N*1e-5(maximum(EIGENS)/minimum(EIGENS)))
b = ones(N)
b̃ = ones(size(Ã, 1))

# ==============================================================================
# Floats on the exact matrix.

function FloatsOnExactMatrix()
    RelErrors = PerformCGFor(A, b, exact=false, epsilon=1e-3) 
    @info "Relative Energy Error for Floats CG on non perturbed matrix"
    RelErrors |> display
    plot(RelErrors, yaxis=:log10)|>display
return nothing end

FloatsOnExactMatrix()

# ==============================================================================
# Exact on the Smeared out matrix. 


function ExactOnSmearedMatrix()
    RelErrors = PerformCGFor(Ã, b̃, exact=true, epsilon=1e-3) 
    @info "Relative Energy Error for exact CG on perturbed matrix"
    RelErrors |> display
    plot(RelErrors, yaxis=:log10)|>display
return nothing end

ExactOnSmearedMatrix()
