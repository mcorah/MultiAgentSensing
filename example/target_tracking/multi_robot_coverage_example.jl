using SubmodularMaximization
using POMDPs
using MCTS
using PyPlot
using Base.Iterators
using Printf
using Statistics

close()

steps = 100
horizon = 5
iterations = 1000

show_coverage = false

print("Enter number of partitions/planning rounds: ")
num_partitions = parse(Int64, readline())

print("Enter number of robots: ")
num_robots = parse(Int64, readline())

num_targets = default_num_targets(num_robots)

grid = Grid(num_robots=num_robots)
sensor = CoverageSensor()

configs = MultiRobotTargetCoverageConfigs(grid=grid,
                                          sensor=sensor,
                                          horizon=horizon)

robot_states = map(x->random_state(grid), 1:num_robots)
target_states = map(x->random_state(grid), 1:num_targets)

solver = x->solve_n_partitions(num_partitions, x, threaded=true)

plot_state_space(grid)
xlim([0, grid.width+1])
ylim([0, grid.height+1])

global plots = []

for ii = 2:steps
  println("Step ", ii)

  prior_robot_states = Array(robot_states)

  @time solution = iterate_target_coverage!(robot_states=robot_states,
                                            target_states=target_states,
                                            configs=configs,
                                            solver=solver)

  global robot_states = solution.robot_states
  global target_states = solution.target_states
  trajectories = solution.trajectories

  #= TODO: Implement and refactor to compute weight matrix

  # Compute weights
  problem = MultiRobotTargetCoverageProblem(robot_states, target_states,
                                            configs)
  @time weight = total_weight(compute_weight_matrix(problem; threaded=true))

  @printf("Weight per robot: %0.3f\n", weight / num_robots )
  =#

  #
  # Plot results
  #

  global plots
  foreach(x->x.remove(), plots)

  plots=[]
  # Plot covered area
  for robot in robot_states
    circle = Circle([robot...], sensor.range_limit)
    append!(plots, SubmodularMaximization.plot_filled_circle(circle, color=:blue, alpha=0.1))
  end

  # Plot robots on top
  for robot in robot_states
    append!(plots, plot_quadrotor(robot, color=:blue))
  end

  for trajectory in trajectories
    append!(plots, plot_states(trajectory, color=:blue, linestyle=":"))
  end

  # Plot motion
  for (prior, current) in zip(prior_robot_states, robot_states)
    append!(plots, plot_states([prior, current], color=:blue, linestyle="-"))
  end

  for target in target_states
    covered = is_covered(sensor, target, robot_states)

    # Dim covered targets
    alpha = covered ? 0.3 : 1.0
    append!(plots, plot_target(target, alpha=alpha))
  end

  #
  # Print a summary of the problem state
  #
  println("Objective: ", solution.objective,
          " (", solution.objective / num_robots, " per robot)")
  n = num_covered(sensor, target_states, robot_states)
  println("Num. covered: ", n, " of ", num_targets, " (", n / num_targets, ")")


  line = readline()

  if line == "q"
    break
  end
end
