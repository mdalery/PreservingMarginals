const TOLERANCE = 1e-8


"""
	are_barycenteric( weights::Vector{Float64} )
	
	Return `true` if weights are positive ans sums to one, `false` else.
"""
@inline function are_barycentric( weights::Vector{Float64} )
	!(false in (weights .> zero(Float64)))  && sum(weights) - one(Float64) < TOLERANCE
end
