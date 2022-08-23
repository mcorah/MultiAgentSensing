# This file provides most of the implementation of multi-robot target *coverage*
# problems.
#
# These coverage problems involve controlling robots to cover a number of
# targets e.g. to keep them within range of onboard sensors.
#
# This problem is applicable to cinematography whereas robots may have good
# estimations of position but otherwise seek to "cover" targets with sensors or
# cameras.

export MultiRobotTargetCoverageConfigs, MultiRobotTargetCoverageProblem

# Stores configuration variables for multi-robot target tracking
struct MultiRobotTargetCoverageConfigs
  grid::Grid
  sensor::CoverageSensor
  horizon::Int64

  # Configuration for the MCTS solver
  solver_iterations::Int64

  robot_target_range_limit::Float64

  function MultiRobotTargetCoverageConfigs(;
                                           grid,
                                           horizon=default_horizon,
                                           sensor=CoverageSensor(),
                                           solver_iterations =
                                           default_num_iterations[horizon],
                                           robot_target_range_limit = Inf
                                          )

    new(grid, sensor, horizon, solver_iterations, robot_target_range_limit)
  end
end

# Constructor that can infer parameters based on the number of robots
#
# Note that inferred parameters can still be overriden, but we probably don't
# want to do so
function MultiRobotTargetCoverageConfigs(num_robots;
                                         kwargs...)
  grid=Grid(num_robots=num_robots)
  MultiRobotTargetCoverageConfigs(;grid=grid, kwargs...)
end

# Copy constructor with overrides
function
  MultiRobotTargetCoverageConfigs(configs::MultiRobotTargetCoverageConfigs;
                                  kwargs...)

  fields = fieldnames(MultiRobotTargetCoverageConfigs)
  old_settings = Dict(field => getfield(configs, field) for field in fields)
  settings = merge(old_settings, kwargs)

  MultiRobotTargetCoverageConfigs(; settings...)
end

# Solution elements consist of the robot index and the associated trajectory
# We construct solutions as such because the output may not have the same
# ordering as the robots
#
# Warning: this "problem" object may become invalid if any of the underlying
# objects change and may copy some but not of the inputs.
struct MultiRobotTargetCoverageProblem <: AbstractMultiRobotProblem
  # Target tracking problems are defined by vectors of robot states
  partition_matroid::Vector{State}

  target_states::Vector{State}

  configs::MultiRobotTargetCoverageConfigs
end

# Construct a target coverage problem with configs.
function MultiRobotTargetCoverageProblem(robot_states::Vector{State},
                                         target_states::Vector{State};
                                         kwargs...)
  configs = MultiRobotTargetCoverageConfigs(;kwargs...)

  MultiRobotTargetCoverageProblem(robot_states, target_states, configs)
end

# Objective function is often used after solving: What was the total objective
# value after running the solver?
# X is an Vector of (Int, Trajectory) tuples
function objective(p::MultiRobotTargetCoverageProblem, X)
  configs = p.configs

  trajectories = map(last, X)

  sum(p.target_states) do target_state
    finite_horizon_coverage(configs.grid, target_state, configs.sensor,
                            trajectories;
                           ).reward
  end
end

# Problems requiring specialized solvers define solve block
function solve_block(p::MultiRobotTargetCoverageProblem, block::Integer,
                     selections::Vector)
  configs = p.configs

  trajectories = map(last, selections)

  problem = SingleRobotTargetCoverageProblem(configs.grid,
                                             configs.sensor,
                                             configs.horizon,
                                             p.target_states,
                                             prior_trajectories=trajectories)

  state = get_state(p, block)

  solution = solve_single_robot(problem, state,
                                n_iterations=configs.solver_iterations)

  (block, solution.trajectory)
end
