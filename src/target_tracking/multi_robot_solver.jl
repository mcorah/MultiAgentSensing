using LinearAlgebra

export MultiRobotTargetTrackingProblem

struct MultiRobotTargetTrackingProblem <: PartitionProblem
  grid::Grid
  sensor::RangingSensor
  horizon::Int64
  target_filters::Vector{Filter{Int64}}

  # Target tracking problems are defined by vectors of robot states
  partition_matroid::Vector{State}

  solver_iterations::Int64
  solver_information_samples::Int64

  # We will generally use a larger number of samples when querying actual
  # solution values
  objective_information_samples::Int64

  function MultiRobotTargetTrackingProblem(grid, sensor, horizon,
                                           target_filters,
                                           robot_states;
                                           solver_iterations =
                                             default_num_iterations,
                                           solver_information_samples =
                                             default_solver_information_samples,
                                           objective_information_samples =
                                             default_num_information_samples
                                          )

    new(grid, sensor, horizon, target_filters, robot_states, solver_iterations,
        solver_information_samples, objective_information_samples)
  end
end

# Solution elements consist of the robot index and the associated trajectory
# We construct solutions as such because the output may not have the same
# ordering as the robots
PartitionElement(::Type{MultiRobotTargetTrackingProblem}) =
  Tuple{Int64, Trajectory}

function objective(p::MultiRobotTargetTrackingProblem, X)
  trajectories = map(last, X)

  sum(p.target_filters) do filter
    finite_horizon_information(p.grid, filter, p.sensor,
                               trajectories;
                               num_samples=p.objective_information_samples
                              ).reward
  end
end

function get_state(p::MultiRobotTargetTrackingProblem, index)
  p.partition_matroid[index]
end

function solve_block(p::MultiRobotTargetTrackingProblem, block::Integer,
                     selections::Vector)

  trajectories = map(last, selections)

  problem = SingleRobotTargetTrackingProblem(p.grid, p.sensor, p.horizon,
                                             p.target_filters,
                                             prior_trajectories=trajectories,
                                             num_information_samples=
                                               p.solver_information_samples
                                            )

  state = get_state(p, block)

  solution = solve_single_robot(problem, state,
                                n_iterations=p.solver_iterations)

  (block, solution.trajectory)
end
