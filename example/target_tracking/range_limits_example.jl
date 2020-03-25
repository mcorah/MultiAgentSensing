# Minimal example that demonstrates robot-robot and robot-target range limits.

using SubmodularMaximization
using POMDPs
using MCTS
using PyPlot
using Base.Iterators
using Printf
using Statistics

close()

steps = 100
horizon = SubmodularMaximization.default_horizon

show_observations = false
sparse = false

num_partitions = 4

communication_range = 20
robot_target_range_limit = 12

print("Enter number of robots: ")
num_robots = parse(Int64, readline())

num_targets = default_num_targets(num_robots)

grid = Grid(num_robots=num_robots)

configs = MultiRobotTargetTrackingConfigs(grid=grid,
                                          horizon=horizon,
                                          robot_target_range_limit=
                                            robot_target_range_limit
                                         )

robot_states = map(x->random_state(grid), 1:num_robots)
target_states = map(x->random_state(grid), 1:num_targets)

if sparse
  global histogram_filters = map(x->SparseFilter(grid, x, threshold=1e-3),
                                 target_states)
else
  global histogram_filters = map(x->Filter(grid, x), target_states)
end

solver = x->solve_communication_range_limit(x, threaded=true,
                                            num_partitions=num_partitions,
                                            communication_range=
                                            communication_range)

plot_state_space(grid)
xlim([0, grid.width+1])
ylim([0, grid.height+1])


for ii = 2:steps
  println("Step ", ii)

  @time solution = iterate_target_tracking!(robot_states=robot_states,
                                            target_states=target_states,
                                            target_filters=histogram_filters,
                                            configs=configs,
                                            solver=solver)

  global robot_states = solution.robot_states
  global target_states = solution.target_states
  trajectories = solution.trajectories
  range_observations = solution.range_observations

  # Compute weights
  problem = MultiRobotTargetTrackingProblem(robot_states, histogram_filters,
                                            configs)
  @time weight = total_weight(compute_weight_matrix(problem; threaded=true))

  @printf("Weight per robot: %0.3f\n", weight / num_robots )

  #
  # Plot results
  #

  plots=[]
  for robot in robot_states
    append!(plots, plot_quadrotor(robot, color=:blue))
  end

  for trajectory in trajectories
    append!(plots, plot_states(trajectory, color=:blue, linestyle=":"))
  end

  if show_observations
    for (robot, observations) in zip(robot_states, range_observations)
      for observation in observations
        append!(plots, plot_observation(robot, observation, color=:blue))
      end
    end
  end

  for target in target_states
    append!(plots, plot_target(target))
  end

  append!(plots, visualize_filters(histogram_filters))

  #
  # Print a summary of the problem state
  #
  println("Objective: ", solution.objective,
          " (", solution.objective / num_robots, " per robot)")
  e = entropy(histogram_filters)
  println("Entropy: ", e, " (", e / num_targets, " per target)")

  foreach(drop_below_threshold!, histogram_filters)

  if sparse
    @printf("Sparsity: %0.2f\n", mean(sparsity, histogram_filters))
  end

  line = readline()

  if line == "q"
    break
  end

  if ii < steps
    foreach(x->x.remove(), plots)
  end
end
