# This file provides most of the implementation of multi-robot target *tracking*
# problems.
#
# The tracking problems involve estimation of target positions and optimization
# via maximizing mutual information.

using LinearAlgebra

export MultiRobotTargetTrackingConfigs, MultiRobotTargetTrackingProblem

# Automatically switch to sparse filters for large numbers of robots
const num_robots_sparse_filtering_threshold = 15
const sparsity_threshold=1e-3

function compute_filter_means(target_filters::Vector{<:AnyFilter})
  map(target_filters) do x
    Tuple{Float64, Float64}(weighted_average(x))
  end
end
