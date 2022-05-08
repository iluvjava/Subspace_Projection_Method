include("Lanczos_Ritz_Utilities.jl")
include("../../src/iterative_lanczos.jl")


function LanczosPojectOnRitz(
    il::IterativeLanczos; 
    offset=20, 
    exact=false
)
    il.reorthogonalize = exact
    n = il.Q[1]|>length
    for _ in 1: offset - 1
        il()
    end
    ProjOnRitz = Vector()
    for II in offset:(n) - 1
        q = il()
        T = il|>GetTMatrix
        Q = il|>GetQMatrix
        s = eigvecs(T)[:, end:-1:(end - 2)]
        push!(ProjOnRitz, q'*Q*s)
    end

return offset:(n-1), vcat(ProjOnRitz...) end


function Prepare()
    n = 64
    A = Diagonal(LinRange(-1, 1, n).^3)
    il = IterativeLanczos(A, ones(n))
    il(n - 1)
    A = il|>GetTMatrix
    display(A)
    e1 = zeros(n)
    e1[1] = 1
    il = IterativeLanczos(A, e1)
return il end

il = Prepare()
xaxis, ys = LanczosPojectOnRitz(il)
fig = plot(
    xaxis, abs.(ys[:, 1]),
    yaxis=:log,
    xlabel="iterations: k", 
    ylabel="\$q_k^TQ_ks_i^{(k)}\$", 
    label="\$q^T_kQ_ks_1\$",
    legend=:bottomleft, size=(750, 500), 
    dpi=300, leftmargin=5*Plots.mm
)

plot!(
    fig, xaxis, abs.(ys[:, 2]),
    linestyle=:dashdot, 
    label="\$q^T_kQ_ks_2\$"
)

plot!(
    fig, xaxis, abs.(ys[:, 3]),
    linestyle=:dash,
    label="\$q^T_kQ_ks_3\$"
)

title!(fig, "Ritz Proj with Lanczs on \$T_k, q_1=\\xi_1\$")
savefig(fig,"$(@__DIR__)/plots/ritz_proj_tridiagonal")
fig|>display