include("Lanczos_Ritz_Utilities.jl")
include("../../src/iterative_lanczos.jl")

### ============================================================================
### Lanczos vector project onto the extreme ritz value's eigenvalues
### 

function PerformAndProject(n=64, offset=10)
    A = Diagonal(LinRange(-1, 1, n).^3)
    il = IterativeLanczos(A, ones(n))
    for _ in 1: offset
        il()
    end
    ProjOnRitz = Vector()
    for _ in offset:n
        q = il()
        T = il|>GetTMatrix
        Q = il|>GetQMatrix
        s = eigvecs(T)[:, end - 2: end]
        push!(ProjOnRitz, q'*Q*s)
    end

return vcat(ProjOnRitz...) end

ProjOnRitz = PerformAndProject()
ProjOnRitz = ProjOnRitz .|> abs
Iters = 10:64
fig1 = plot(
    Iters, 
    ProjOnRitz[:, 1], 
    yaxis=:log, 
    legend=:bottomleft, size=(750, 500), 
    label="\$q^T_kQ_ks_1\$"
    )
plot!(fig1, Iters, ProjOnRitz[:, 2], linestyle=:dashdot, label="\$q^T_kQ_ks_2\$")
plot!(fig1, Iters, ProjOnRitz[:, 3], linestyle=:dash, label="\$q^T_kQ_ks_3\$")
title!("Lanczos Vector Project onto Ritz Vectors")
yaxis!("\$\\log(|q^T_kQS|)\$")
xaxis!("iterations") 
fig1|>display
savefig(fig1, "$(@__DIR__)/plots/lanczos_proj_on_ritz.png")

