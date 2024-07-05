"""
	Slater <: Atom
	Slater(ζ::Float64, r::Float64)

	Definition of an atomic Slater distribution with positive scale parameter `ζ` and position parameter `r`.
"""
struct Slater <: Atom{1}
	ζ::Float64
	r::Float64

	function Slater(ζ::Float64, r::Float64)
		@assert ζ > zero(Float64)

		new(ζ, r)
	end
end


"""
	function ( atom::Slater )( x::Float64 )

	Distribution function of a Slater distribution.
"""
@inline function ( atom::Slater )( x::Vararg{Float64, 1} )
    atom.ζ/2 * exp(-atom.ζ * abs(x - atom.r[1]))
end


"""
	atomic_distance( atom1::Slater, atom2::Slater )

	Compute the squared atomic distance between `atom1` and `atom2`.
"""
@inline function sq_atomic_distance( atom1::Slater, atom2::Slater )
	( atom1.r - atom2.r )^2 + 2*( 1/atom1.ζ - 1/atom2.ζ )^2
end

"""
	barycenter( atoms::Vector{Slater}, λ::Vector{Float64} )
	
	Compute the barycenter between atoms `atoms` with weights `λ`.
"""
@inline function barycenter( atoms::Vector{Slater}, λ::Vector{Float64} )
	@assert length(atoms) == length(λ)

	ζ = 1 / sum(λ ./ [ atom.ζ for atom in atoms ])
	r = sum(λ .* [ atom.r for atom in atoms])
	Slater(ζ, r)
end


"""
	MixtureSlater( ζs::Vector{Float64}, rs::Vector{Float64}, weights::Vector{Float64} )

	Create a `Mixture{Slater}` object with scales parameters `ζs`, positions parameters `rs` and weights `weights`.
"""
@inline function MixtureSlater( ζs::Vector{Float64}, rs::Vector{Float64}, weights::Vector{Float64} )
	Mixture([ Slater(ζ, r) for (ζ, r) in collect(zip(ζs, rs)) ], weights)
end


"""
	MixtureSlater( ζs::Vector{Float64}, rs::Vector{Float64}, weights::Vector{Float64} )

	Create a `Mixture{Slater}` object with same scales parameters `ζ`, positions parameters `rs` and weights `weights`.
"""
@inline function MixtureSlater( ζ::Float64, rs::Vector{Float64}, weights::Vector{Float64} )
	Mixture([ Slater(ζ, r) for r in rs ], weights)
end
