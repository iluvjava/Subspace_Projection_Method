include("./utilities.jl")
include("../../src/CGPO.jl")

Eigens = EigDistribution(rho=0.9)
A, Ã = TinyIntervalTestMatrices(Eigens, scale=maximum(Eigens)/minimum(Eigens))

# ==============================================================================
# Floats on the exact matrix.

function FloatsOnExactMatrix()

return end

# ==============================================================================
# Exact on the Smeared out matrix. 


function ExactOnSmearedMatrix()
    
return end


