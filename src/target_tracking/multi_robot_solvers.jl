using LinearAlgebra

abstract type AbstractTargetProblem <: PartitionProblem{Tuple{Int64,
                                                              Trajectory}}
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

function get_state(p::AbstractTargetProblem, index)
  p.partition_matroid[index]
end

# Return agent center (which is just the state)
# The original coverage code involved more complex agent representations.
get_center(x::Tuple{Int64,Int64}) = [x...]

# Returns distance from robot state to target mean
# (due to convexity and Jensen's inequality, this is a lower bound on the range)
function target_mean_distance(p::MultiRobotTargetTrackingProblem,
                     robot_state::State, target_index::Integer)
  norm(robot_state .- p.filter_means[target_index])
end
function target_mean_distance(p::MultiRobotTargetTrackingProblem,
                     robot_index::Integer, target_index::Integer)
  target_mean_distance(p, p.partition_matroid[robot_index], target_index)
end

function filter_targets_in_range(p::MultiRobotTargetTrackingProblem,
                                 robot_index::Integer;
                                 range_limit=p.configs.robot_target_range_limit
                                )

  # filter by range, producing tuples
  in_bounds = filter(1:length(p.filter_means)) do target_index
    target_mean_distance(p, robot_index, target_index) < range_limit
  end

  map(x->p.target_filters[x[1]], in_bounds)
end

function solve_block(p::MultiRobotTargetTrackingProblem, block::Integer,
                     selections::Vector)
  configs = p.configs

  trajectories = map(last, selections)

  # Select filters in range
  local targets_in_range
  if p.configs.robot_target_range_limit == Inf
    targets_in_range = p.target_filters
  else
    targets_in_range = filter_targets_in_range(p, block)
  end

  problem = SingleRobotTargetTrackingProblem(configs.grid, configs.sensor,
                                             configs.horizon, targets_in_range,
                                             prior_trajectories=trajectories,
                                             num_information_samples=
                                             configs.solver_information_samples
                                            )

  state = get_state(p, block)

  solution = solve_single_robot(problem, state,
                                n_iterations=configs.solver_iterations)

  (block, solution.trajectory)
end

function sample_block(p::AbstractTargetProblem, block::Integer)
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
