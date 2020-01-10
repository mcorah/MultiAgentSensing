using LinearAlgebra

export MultiRobotTargetTrackingConfigs, MultiRobotTargetTrackingProblem

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
                                           horizon,
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

# Solution elements consist of the robot index and the associated trajectory
# We construct solutions as such because the output may not have the same
# ordering as the robots
struct MultiRobotTargetTrackingProblem <: PartitionProblem{Tuple{Int64,
                                                                 Trajectory}}
  # Target tracking problems are defined by vectors of robot states
  partition_matroid::Vector{State}

  target_filters::Vector{Filter{Int64}}

  configs::MultiRobotTargetTrackingConfigs

  function MultiRobotTargetTrackingProblem(robot_states::Vector{State},
                                       target_filters::Vector{<:Filter},
                                       configs::MultiRobotTargetTrackingConfigs)
    new(robot_states, target_filters, configs)
  end
end

# Construct a target tracking problem with configs. I may or may not need this
function MultiRobotTargetTrackingProblem(robot_states::Vector{State},
                                         target_filters::Vector{<:Filter};
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
