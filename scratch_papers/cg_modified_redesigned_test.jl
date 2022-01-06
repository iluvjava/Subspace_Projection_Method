include("cg_modified_redesigned.jl")
using BenchmarkTools, Profile

n = 1024
A = Diagonal(collect(LinRange(1e-3, 1, n))).^2
b = randn(n)
bNorm = norm(b)
x0 = zeros(n); x0[1] = 1
function trial1()
    cgm = ConjGradModified(A, b, x0)
    ResNorm = norm(cgm.r)
    cgm.storage_limit = n
    cgm.ortho_period = 1024
    cgm.reorthogonalize = true
    iterationCount = 0
    while ResNorm > 1e-10
        ResNorm = cgm()/bNorm
        iterationCount += 1
        # println(ResNorm)
        if ResNorm == Inf || ResNorm == NaN
            error("Resnorm is inf or nan.")
        end
    end
    println("iterationCount: $iterationCount")
    println("Final Error: $(norm(b - A*cgm.x))")
end

function trial2()
    cgm = ConjGradModified(A, b, x0)
    ResNorm = norm(cgm.r)
    cgm.storage_limit = 256
    cgm.reorthogonalize = false
    iterationCount = 0
    while ResNorm > 1e-10
        ResNorm = cgm()/bNorm
        iterationCount += 1
        # println(ResNorm)
        if ResNorm == Inf || ResNorm == NaN
            error("Resnorm is inf or nan.")
        end
    end
    # println("iterationCount: $iterationCount")
end

@info "with Loss of Orthogonality check and re-orthogonalization: "
display(@time trial1())
# @info "without loss of Orthogonality check and re-orthogonalization: "
# display(@benchmark trial2())