using SubmodularMaximization
using POMDPs
using MCTS
using PyPlot
using Base.Iterators

close()

steps = 100
horizon = 5
show_observations = false

num_partitions = 4

print("Enter number of robots: ")
num_robots = parse(Int64, readline())

num_targets = default_num_targets(num_robots=num_robots)

grid = Grid(num_robots=num_robots)
sensor = RangingSensor()

robot_states = map(x->random_state(grid), 1:num_robots)
target_states = map(x->random_state(grid), 1:num_targets)

histogram_filters = map(x->Filter(grid, x), target_states)

plot_state_space(grid)
xlim([0, grid.width+1])
ylim([0, grid.height+1])

for ii = 2:steps
  println("Step ", ii)

  #
  # Update robot and target states
  #

  # Solve for robot and update state
  problem = MultiRobotTargetTrackingProblem(robot_states,
                                            histogram_filters,
                                            grid=grid,
                                            sensor=sensor,
                                            horizon=horizon)

  @time solution = solve_n_partitions(num_partitions, problem, threaded=true)

  for (index, trajectory) in solution.elements
    robot_states[index] = trajectory[1]
  end
  trajectories = map(last, solution.elements)


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
      generate_observation(grid, sensor, robot, target)
    end
  end

  # Measurement update
  for (robot, observations) in zip(robot_states, range_observations)
    for (filter, observation) in zip(histogram_filters, observations)
      measurement_update!(filter, robot, get_states(grid), sensor, grid,
                          observation)
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
