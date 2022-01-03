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

# Solution elements consist of the robot index and the associated trajectory
# We construct solutions as such because the output may not have the same
# ordering as the robots
#
# Warning: this "problem" object may become invalid if any of the underlying
# objects change and may copy some but not of the inputs.
struct MultiRobotTargetTrackingProblem{F<:AnyFilter} <: AbstractTargetProblem
  # Target tracking problems are defined by vectors of robot states
  partition_matroid::Vector{State}

  target_filters::Vector{F}
  filter_means::Vector{Tuple{Float64, Float64}}

  configs::MultiRobotTargetTrackingConfigs

  function MultiRobotTargetTrackingProblem(robot_states::Vector{State},
                 target_filters::Vector{F},
                 configs::MultiRobotTargetTrackingConfigs) where F <: AnyFilter

    if length(robot_states) > num_robots_sparse_filtering_threshold
      sparse_filters = map(x->SparseFilter(x, threshold=sparsity_threshold),
                           target_filters)

      filter_means = compute_filter_means(sparse_filters)

      new{SparseFilter{Int64}}(robot_states, sparse_filters, filter_means,
                               configs)
    else
      filter_means = compute_filter_means(target_filters)

      new{F}(robot_states, target_filters, filter_means, configs)
    end
  end
end

# Construct a target tracking problem with configs. I may or may not need this
function MultiRobotTargetTrackingProblem(robot_states::Vector{State},
                                         target_filters::Vector{<:AnyFilter};
                                         kwargs...)
  configs = MultiRobotTargetTrackingConfigs(;kwargs...)
  MultiRobotTargetTrackingProblem(robot_states, target_filters, configs)
end

function compute_filter_means(target_filters::Vector{<:AnyFilter})
  map(target_filters) do x
    Tuple{Float64, Float64}(weighted_average(x))
  end
end
