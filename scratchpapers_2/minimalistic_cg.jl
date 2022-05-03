"""
    The bare minimum algorithm of CG for forward error analysis.
    It's for the forward error analysis package. 
"""
function cgs(
        A::AbstractMatrix, 
        b::Abstractvector; 
        x0=nothing, 
        k::Int=nothing
    )
    if k === nothing
        k = length(b)
    end
    if x0 === nothing
        x0 = ones(b|>length)
    end
    r = b - A*x0
    x = x0
    p = r
    for II = 1:k
        Ap = A*p
        a = dot(r, r)/dot(p, Ap)
        b = 1/dot(r, r)
        x += x*p
        r -= -a*Ap
        b *= dot(r, r)
        p = r + b*p
    end
    return x
end





