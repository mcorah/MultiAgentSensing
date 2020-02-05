using SubmodularMaximization
using POMDPs
using MCTS
using PyPlot
using Printf

close()

steps = 100
grid_size = 10
horizon = SubmodularMaximization.default_horizon
iterations = 1000

grid = Grid(grid_size, grid_size)
sensor = RangingSensor()

robot_state = random_state(grid)
target_state = random_state(grid)

target_states = Array{State}(undef, steps)
target_states[1] = target_state

robot_states = Array{State}(undef, steps)
robot_states[1] = robot_state

histogram_filter = SparseFilter(grid, target_state, threshold=1e-3)

problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                           [histogram_filter])

plot_state_space(grid)
xlim([0, grid_size+1])
ylim([0, grid_size+1])

for ii = 2:steps
  drop_below_threshold!(histogram_filter)

  println("Step ", ii)
  @printf("Sparsity: %0.2f\n", sparsity(histogram_filter))

  # Before the target moves and the robot receives a measurement, execute robot
  # dynamics
  @time solution = solve_single_robot(problem, robot_state,
                                      n_iterations = iterations)

  global robot_state = solution.action
  trajectory = solution.trajectory
  robot_states[ii] = robot_state

  # Then update the target and sample the observation
  global target_state = target_dynamics(grid, target_states[ii-1])
  target_states[ii] = target_state

  range_observation = generate_observation(grid, sensor, robot_state,
                                           target_state)

  # After updating states, update the filter (in place for now)
  process_update!(histogram_filter, transition_matrix(grid))
  measurement_update!(histogram_filter, robot_state,
                      get_states(histogram_filter), sensor, grid,
                      range_observation)

  plots=[]
  append!(plots, plot_trajectory(robot_states[1:ii], color=:blue))
  append!(plots, plot_states(trajectory, color=:blue, linestyle=":"))
  append!(plots, plot_observation(robot_state, range_observation, color=:blue))
  append!(plots, plot_trajectory(target_states[1:ii]))
  append!(plots, visualize_filter(histogram_filter))

  line = readline()

  if line == "q"
    break
  end

  if ii < steps
    foreach(x->x.remove(), plots)
  end
end
