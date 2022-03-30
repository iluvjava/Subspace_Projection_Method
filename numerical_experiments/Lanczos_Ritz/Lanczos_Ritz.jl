include("Lanczos_Ritz_Utilities.jl")
include("../../src/iterative_lanczos.jl")

# ------------------------------------------------------------------------------
# Demonstrating the loss of orthogonality of the lanczos vectors. 
# And the linear dependence of eigenvalues of the exterior of the spectrum. 
# ------------------------------------------------------------------------------
n = 128
A = Diagonal(LinRange(1e-3, 1, n).^2)
il = IterativeLanczos(A, rand(n))
for _ in 1: n - 1
    il()
end
Q = GetQMatrix(il)
fig = heatmap(Q'*Q .|> abs .|> log2, size=(1024, 1024))
ToPlot = Q'*Q .|> abs .|> log2
fig2 = heatmap(ToPlot .>= -10)
mkpath("$(@__DIR__)/plots")
λ, S = eigen(GetTMatrix(il))
Y = Q*S
fig3 = heatmap(Y'*Y .|> abs .|> log2)
savefig(fig, "$(@__DIR__)/plots/fig.png")
savefig(fig2, "$(@__DIR__)/plots/fig2.png")
savefig(fig3, "$(@__DIR__)/plots/fig3.png")

# ------------------------------------------------------------------------------
# All the ritz values during the computations process. 
# ------------------------------------------------------------------------------

A = Diagonal(LinRange(1e-3, 1, n))
il = IterativeLanczos(A, rand(n))
FoundRitzValues = Vector{Vector{Float64}}()
push!(FoundRitzValues, [GetTMatrix(il)])
TrueEigenValues = diag(A)
for _ in 1: n - 1
    il()
    T = GetTMatrix(il)
    λs, _ = eigen(T)
    push!(FoundRitzValues, λs)
end


for RitzValue in FoundRitzValues
    sort!(RitzValue, rev=true)
end

fig4 = scatter(title="Ritz Values During Iterations", legend=false)
for Idx in 1: 10
    RitzTrajectory = Vector{Float64}()
    for RitzValues in FoundRitzValues
        if Idx <= length(RitzValues)
            push!(RitzTrajectory, RitzValues[Idx])
        end
    end
    scatter!(
        fig4, 
        Idx: length(RitzTrajectory) + Idx - 1, 
        RitzTrajectory, size=(1000,1500), 
        dpi=250, 
        markershape=:cross
    )
end
for Idx in 1:10
    RitzTrajectory = Vector{Float64}()
    for RitzValues in FoundRitzValues
        if Idx <= length(RitzValues)
            push!(RitzTrajectory, RitzValues[end - Idx + 1])
        end
    end
    scatter!(
        fig4, 
        Idx: length(RitzTrajectory) + Idx - 1, 
        RitzTrajectory, size=(1000,1500), 
        dpi=250, 
        markershape=:xcross
    )
end

display(fig4)
savefig(fig4, "$(@__DIR__)/plots/fig4.png")

T = GetTMatrix(il)