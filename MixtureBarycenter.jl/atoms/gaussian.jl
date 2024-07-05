"""
    Gaussian{N} <: Atom
    Gaussian(Σ::Matrix{Float64}, r::Vector{Float64})

	Definition of an atomic multidimensional Gaussian distribution with definite positive scale parameter `Σ` and position parameter `r`.
"""
struct Gaussian{N} <: Atom{N}
    Σ::Matrix{Float64}
    r::Vector{Float64}

    function Gaussian( Σ::Matrix{Float64}, r::Vector{Float64} )
        N = length(r)
        @assert size(Σ) == (N, N)
        @assert isposdef(Σ)

        new{N}(Σ, r)
	end
end


"""
    function ( atom::Gaussian{N} )( x::Vararg{Float64, N} ) where N

	Distribution function of a multidimensional Gaussian distribution.
"""
@inline function ( atom::Gaussian{N} )( x::Vararg{Float64, N} ) where N
    x = [ a for a in x ]
    exp(-0.5 * (x - atom.r)' * inv(atom.Σ) * (x - atom.r)) / ( (2*π)^(N/2) * sqrt(det(atom.Σ)) )
end


"""
    atomic_distance( atom1::Gaussian{N}, atom2::Gaussian{N} ) where N

	Compute the squared atomic distance between `atom1` and `atom2`.
"""
@inline function sq_atomic_distance( atom1::Gaussian{N}, atom2::Gaussian{N} ) where N
    sum( (x1 - x2)^2 for (x1, x2) in zip(atom1.r, atom2.r) ) + tr(atom1.Σ + atom2.Σ - 2*sqrt(sqrt(atom1.Σ) * atom2.Σ * sqrt(atom1.Σ)))
end


"""
    barycenter( atoms::Vector{Gaussian{N}}, λ::Vector{Float64} ) where N
	
	Compute the barycenter between atoms `atoms` with weights `λ`.
"""
@inline function barycenter( atoms::Vector{Gaussian{N}}, λ::Vector{Float64} ) where N
    @assert length(atoms) == length(λ)

    Σ = I(N)
    for _ in 1:15
        Σ = sum( λ[i] * sqrt(sqrt(Σ) * atoms[i].Σ * sqrt(Σ)) for i in 1:length(λ) )
    end
    r = sum(λ .* [ atom.r for atom in atoms ])
    Gaussian(Matrix(Hermitian(Σ)), r)
end


"""
    function MixtureGaussian( Σs::Vector{Matrix{Float64}}, rs::Vector{Vector{Float64}}, weights::Vector{Float64} )

    Create a `Mixture{Gauassian{N}}` object with scales parameters `Σs`, positions parameters `rs` and weights `weights`.
"""
@inline function MixtureGaussian( Σs::Vector{Matrix{Float64}}, rs::Vector{Vector{Float64}}, weights::Vector{Float64} )
    @assert all( length(r) == length(rs[1]) for r in rs[2:end] )
    Mixture([ Gaussian(Σ, r) for (Σ, r) in collect(zip(Σs, rs)) ], weights)
end
