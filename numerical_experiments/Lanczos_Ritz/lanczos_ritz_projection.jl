include("Lanczos_Ritz_Utilities.jl")
include("../../src/iterative_lanczos.jl")

### ============================================================================
### Lanczos vector project onto the extreme ritz value's eigenvalues
### 

function PerformAndProject(n=64; offset=20, exact=false)
    A = Diagonal(LinRange(-1, 1, n).^3)
    il = IterativeLanczos(A, ones(n))
    il.reorthogonalize = exact
    for _ in 1: offset - 1
        il()
    end
    ProjOnRitz = Vector()
    for II in offset:n - 1
        q = il()
        T = il|>GetTMatrix
        Q = il|>GetQMatrix
        
        s = eigvecs(T)[:, end:-1: (end - 2)]
        push!(ProjOnRitz, q'*Q*s)
    end

return vcat(ProjOnRitz...) end

# ----- Floats --------
ProjOnRitz = PerformAndProject()
ProjOnRitz = ProjOnRitz .|> abs
Iters = 20:63
fig1 = plot(
    Iters, 
    ProjOnRitz[:, 1], 
    yaxis=:log, 
    legend=:bottomleft, size=(750, 500), 
    label="\$q^T_kQ_ks_1\$", 
    dpi=300
)
plot!(fig1, Iters, ProjOnRitz[:, 2], linestyle=:dashdot, label="\$q^T_kQ_ks_2\$")
plot!(fig1, Iters, ProjOnRitz[:, 3], linestyle=:dash, label="\$q^T_kQ_ks_3\$")
title!("Lanczos vec proj onto Ritz Vectors (Floats)")
yaxis!("\$\\log(|q^T_kQS|)\$")
xaxis!("iterations") 
fig1|>display
savefig(fig1, "$(@__DIR__)/plots/lanczos_proj_on_ritz_float.png")

# ------ Exact --------
ProjOnRitz = PerformAndProject(exact=true)
ProjOnRitz = ProjOnRitz .|> abs
Iters = 20:63
fig1 = plot(
    Iters, 
    ProjOnRitz[:, 1], 
    yaxis=:log, 
    legend=:bottomleft, size=(750, 500), 
    label="\$q^T_kQ_ks_1\$", 
    dpi=300
)

plot!(fig1, Iters, ProjOnRitz[:, 2], linestyle=:dashdot, label="\$q^T_kQ_ks_2\$")
plot!(fig1, Iters, ProjOnRitz[:, 3], linestyle=:dash, label="\$q^T_kQ_ks_3\$")
title!("Lanczos vec proj on ritz vectors (largest 3) (Exact)")
yaxis!("\$\\log(|q^T_kQS|)\$")
xaxis!("iterations") 
fig1|>display
savefig(fig1, "$(@__DIR__)/plots/lanczos_proj_on_ritz_exact.png")
