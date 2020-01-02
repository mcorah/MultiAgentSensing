using SubmodularMaximization
using POMDPs
using MCTS
using PyPlot
using Base.Iterators

close()

steps = 100
horizon = 5
solver_iterations = 1000
grid_cells_per_robot = 50
show_observations = false

print("Enter number of robots: ")
num_robots = parse(Int64, readline())

num_targets = div(num_robots, 2)

grid_size = round(Int64, sqrt(grid_cells_per_robot * num_robots))

grid = Grid(grid_size, grid_size)
sensor = RangingSensor(0.5^2, 0.1^2)

robot_states = map(x->random_state(grid), 1:num_robots)
target_states = map(x->random_state(grid), 1:num_targets)

histogram_filters = map(x->Filter(grid, x), target_states)

# got this far

plot_state_space(grid)
xlim([0, grid_size+1])
ylim([0, grid_size+1])

for ii = 2:steps
  println("Step ", ii)

  #
  # Update robot and target states
  #

  # Solve for robot and update state
  problem = MultiRobotTargetTrackingProblem(grid, sensor, horizon,
                                            histogram_filters, robot_states,
                                            solver_iterations=solver_iterations
                                           )

  @time solution = solve_sequential(problem)
  trajectories = solution.elements

  global robot_states = map(first, trajectories)


  # Update Target states
  global target_states = map(x->target_dynamics(grid, x), target_states)


  #
  # Run filter updates (corresponding to new target states) and provide robots
  # with observations
  #

  # Process update
  for filter in histogram_filters
    process_update!(filter, transition_matrix(grid))
  end

  # Sample ranging observations
  # (output is an array of arrays of robots' observations)
  range_observations = map(robot_states) do robot
    map(target_states) do target
      generate_observation(sensor, robot, target)
    end
  end

  # Measurement update
  for (robot, observations) in zip(robot_states, range_observations)
    for (filter, observation) in zip(histogram_filters, observations)
      measurement_update!(filter, robot, get_states(grid), sensor, observation)
    end
  end

  #
  # Plot results
  #

  plots=[]
  for robot in robot_states
    append!(plots, plot_trajectory([robot], color=:blue))
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
    append!(plots, plot_trajectory([target]))
  end

  append!(plots, visualize_filters(histogram_filters))

  line = readline()

  if line == "q"
    break
  end

  if ii < steps
    foreach(x->x.remove(), plots)
  end
end
