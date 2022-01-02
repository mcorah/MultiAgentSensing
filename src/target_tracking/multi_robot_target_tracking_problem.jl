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

# Stores configuration variables for multi-robot target tracking
struct MultiRobotTargetTrackingConfigs
  grid::Grid
  sensor::RangingSensor
  horizon::Int64

  # Configuration for the MCTS solver
  solver_iterations::Int64
  solver_information_samples::Int64

  # We will generally use a larger number of samples when querying actual
  # solution values
  objective_information_samples::Int64

  robot_target_range_limit::Float64

  function MultiRobotTargetTrackingConfigs(;
                                           grid,
                                           horizon=default_horizon,
                                           sensor=RangingSensor(),
                                           solver_iterations =
                                           default_num_iterations[horizon],
                                           solver_information_samples =
                                             default_solver_information_samples,
                                           objective_information_samples =
                                             default_num_information_samples,
                                           robot_target_range_limit = Inf
                                          )

    new(grid, sensor, horizon, solver_iterations, solver_information_samples,
        objective_information_samples, robot_target_range_limit)
  end
end

# Constructor that can infer parameters based on the number of robots
#
# Note that inferred parameters can still be overriden, but we probably don't
# want to do so
function MultiRobotTargetTrackingConfigs(num_robots;
                                         kwargs...)
  grid=Grid(num_robots=num_robots)
  MultiRobotTargetTrackingConfigs(;grid=grid, kwargs...)
end

# Copy constructor with overrides
function
  MultiRobotTargetTrackingConfigs(configs::MultiRobotTargetTrackingConfigs;
                                  kwargs...)

  fields = fieldnames(MultiRobotTargetTrackingConfigs)
  old_settings = Dict(field => getfield(configs, field) for field in fields)
  settings = merge(old_settings, kwargs)

  MultiRobotTargetTrackingConfigs(; settings...)
end

function compute_filter_means(target_filters::Vector{<:AnyFilter})
  map(target_filters) do x
    Tuple{Float64, Float64}(weighted_average(x))
  end
end
