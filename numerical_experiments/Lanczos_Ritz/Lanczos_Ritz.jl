include("Lanczos_Ritz_Utilities.jl")
include("../../src/iterative_lanczos.jl")

n = 1024
A = Diagonal(LinRange(1e-3, 1, n))
#A = Diagonal(rand(n))
il = IterativeLanczos(A, rand(n))
for _ in 1: n - 1
    il()
end
Q = GetQMatrix(il)
heatmap(Q'*Q .|> abs .|> log2)