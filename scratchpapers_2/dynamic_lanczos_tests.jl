include("dynamic_symtridiagonal.jl")
include("dynamic_lanczos.jl")
using LinearAlgebra, Plots


n = 32
m = 10
A = Diagonal(LinRange(1e-3, 1, n).^3)
il = DIL(A, ones(n), n, true)
ChangeVelocityTol!(il.dynamic_symtridiagonal, 0.0)
fig1 = plot(legend=false)
for II in m:-1:1
    il()
    Ritzs = GetRitzValues(il)
    ConvergedRitzs = GetConverged(il.dynamic_symtridiagonal)
    scatter!(fig1, Ritzs, II*(Ritzs|>length|>ones), marker=:x)
    scatter!(
        fig1, 
        ConvergedRitzs, 
        II*(ConvergedRitzs|>length|>ones), 
        marker=:square
    )
end
display(fig1)

