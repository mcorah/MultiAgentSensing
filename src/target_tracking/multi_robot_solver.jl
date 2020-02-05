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

  function MultiRobotTargetTrackingConfigs(;
                                           grid,
                                           horizon=default_horizon,
                                           sensor=RangingSensor(),
                                           solver_iterations =
                                           default_num_iterations[horizon],
                                           solver_information_samples =
                                             default_solver_information_samples,
                                           objective_information_samples =
                                             default_num_information_samples
                                          )

    new(grid, sensor, horizon, solver_iterations, solver_information_samples,
        objective_information_samples)
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
struct MultiRobotTargetTrackingProblem{F<:AnyFilter} <: PartitionProblem{Tuple{Int64,
                                                                 Trajectory}}
  # Target tracking problems are defined by vectors of robot states
  partition_matroid::Vector{State}

  target_filters::Vector{F}

  configs::MultiRobotTargetTrackingConfigs

  function MultiRobotTargetTrackingProblem(robot_states::Vector{State},
                 target_filters::Vector{F},
                 configs::MultiRobotTargetTrackingConfigs) where F <: AnyFilter
    if length(robot_states) > num_robots_sparse_filtering_threshold
      sparse_filters = map(x->SparseFilter(x, threshold=sparsity_threshold),
                           target_filters)

      new{SparseFilter{Int64}}(robot_states, sparse_filters, configs)
    else
      new{F}(robot_states, target_filters, configs)
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

function objective(p::MultiRobotTargetTrackingProblem, X)
  configs = p.configs

  trajectories = map(last, X)

  sum(p.target_filters) do filter
    finite_horizon_information(configs.grid, filter, configs.sensor,
                               trajectories;
                               num_samples=configs.objective_information_samples
                              ).reward
  end
end

function get_state(p::MultiRobotTargetTrackingProblem, index)
  p.partition_matroid[index]
end

function solve_block(p::MultiRobotTargetTrackingProblem, block::Integer,
                     selections::Vector)
  configs = p.configs

  trajectories = map(last, selections)

  problem = SingleRobotTargetTrackingProblem(configs.grid, configs.sensor,
                                           configs.horizon, p.target_filters,
                                           prior_trajectories=trajectories,
                                           num_information_samples=
                                             configs.solver_information_samples
                                           )

  state = get_state(p, block)

  solution = solve_single_robot(problem, state,
                                n_iterations=configs.solver_iterations)

  (block, solution.trajectory)
end

function sample_block(p::MultiRobotTargetTrackingProblem, block::Integer)
  horizon = p.configs.horizon
  grid = p.configs.grid

  trajectory = Array{State}(undef, horizon)

  # (target dynamics are the same as random rollouts for tracking robots)
  current_state = p.partition_matroid[block]
  for ii in 1:horizon
    current_state = target_dynamics(grid, current_state)
    trajectory[ii] = current_state
  end

  (block, trajectory)
end
