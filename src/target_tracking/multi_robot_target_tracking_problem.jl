# This file provides most of the implementation of multi-robot target *tracking*
# problems.
#
# The tracking problems involve estimation of target positions and optimization
# via maximizing mutual information.

using LinearAlgebra

function compute_filter_means(target_filters::Vector{<:AnyFilter})
  map(target_filters) do x
    Tuple{Float64, Float64}(weighted_average(x))
  end
end
