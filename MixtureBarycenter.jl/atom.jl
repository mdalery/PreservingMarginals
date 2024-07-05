"""
    Atom{N}

    Abstract type of distribution family for atoms (of dimension N) in mixtures.
	When defining a new Atom subtype A, define
        - its distribution function as ( ::A )( Varargs{N, Float64} )
		- the square distance as sq_atomic_distance( ::A, ::A )
		- the barycenters as barycenter( ::Vector{A}, ::Vector{Float64} )
"""
abstract type Atom{N} end


"""
    dimension( atom::Atom{N} ) where N

    Get the dimension of an Atom object.
"""
function dimension( atom::Atom{N} ) where N
    @assert N isa Int64
    return N
end


"""
	atomic_distance( atom1::A, atom2::A ) where A<:Atom

	Compute the atomic distance between `atom1` and `atom2`.
	Define first sq_atomic_distance for `A`.
"""
@inline function atomic_distance( atom1::A, atom2::A ) where A<:Atom
	sqrt(sq_atomic_distance(atom1, atom2 ))
end


"""
	sq_atomic_mm( atoms::Vector{<:Atom}, λ::Vector{Float64} )

	Compute the squared multi-marginal problem between `atoms` with weights `λ`.
	Define first sq_atomic_distance for the `Atom` subtype used.
"""
@inline function sq_atomic_mm( atoms::Vector{<:Atom}, λ::Vector{Float64} )
	@assert length(atoms) == length(λ)

	bar = barycenter(atoms, λ)
	sum(λ .* [ sq_atomic_distance(atom, bar) for atom in atoms ])
end


"""
	sq_atomic_mm( atoms::Vector{<:Atom} )

	Compute the squared multi-marginal problem between `atoms` with equal weights.
	Define first sq_atomic_distance for the `Atom` subtype used.
"""
@inline function sq_atomic_mm( atoms::Vector{<:Atom} )
	sq_atomic_mm(atoms, 1/length(atoms) * ones(length(atoms)))
end


"""
	sq_atomic_mm( atoms::Vector{<:Atom}, λ::Vector{Float64} )

	Compute the multi-marginal problem between `atoms` with weights `λ`.
	Define first sq_atomic_distance for the `Atom` subtype used.
"""
@inline function atomic_mm( atoms::Vector{<:Atom}, λ::Vector{Float64} )
	sqrt(sq_atomic_mm(atoms, λ))
end


"""
	sq_atomic_mm( atoms::Vector{<:Atom} )

	Compute the multi-marginal problem between `atoms` with equally distributed weights.
	Define first sq_atomic_distance for the `Atom` subtype used.
"""
@inline function atomic_mm( atoms::Vector{<:Atom} )
	sqrt(sq_atomic_mm(atoms))
end
