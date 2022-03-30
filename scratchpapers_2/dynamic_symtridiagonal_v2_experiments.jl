include("dynamic_symtridiagonal_v2.jl")

function CharPolyEvolution(n=10)
    mainDiag = rand(n)
    subDiag = rand(n - 1)
    dynamicT = DynamicSymTridiagonal(mainDiag[1])
    Grid = LinRange(-1, 1, 101) |> collect
    fig = plot(legend=false)
    for Idx in 1:n - 1
        dynamicT(mainDiag[Idx + 1], subDiag[Idx])
        p = Grid .|>(x) -> CharPolyShifted(dynamicT, x) 
        plot!(fig, Grid, sign.(p).*log10.(abs.(p) .+ 1)) 
    end
    display(fig)
return end

CharPolyEvolution()

function MakeMeRandDynamicTridiagonal(n=10)
    mainDiag = rand(n)
    subDiag = rand(n - 1)
    dynamicT = DynamicSymTridiagonal(mainDiag[1])
    for Idx in 1:n - 1
        dynamicT(mainDiag[Idx + 1], subDiag[Idx]) 
    end
return dynamicT end

DymT = MakeMeRandDynamicTridiagonal()
