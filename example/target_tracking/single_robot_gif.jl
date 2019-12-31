using SubmodularMaximization
using POMDPs
using MCTS
using PyPlot

pygui(false)

steps = 20
grid_size = 10
horizon = 2
iterations = 100

grid = Grid(grid_size, grid_size)
sensor = RangingSensor(0.5^2, 0.1^2)

robot_state = random_state(grid)
target_state = random_state(grid)

target_states = Array{State}(undef, steps)
target_states[1] = target_state

robot_states = Array{State}(undef, steps)
robot_states[1] = robot_state

histogram_filter = Filter(grid, target_state)

problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                           [histogram_filter])

plot_state_space(grid)
xlim([0, grid_size+1])
ylim([0, grid_size+1])

for ii = 2:steps
  println("Step ", ii)

  # Before the target moves and the robot receives a measurement, execute robot
  # dynamics
  global robot_state = solve_single_robot(problem, robot_state,
                                          n_iterations = iterations)
  robot_states[ii] = robot_state

  # Then update the target and sample the observation
  global target_state = target_dynamics(grid, target_states[ii-1])
  target_states[ii] = target_state

  range_observation = generate_observation(sensor, robot_state, target_state)

  # After updating states, update the filter (in place for now)
  process_update!(histogram_filter, transition_matrix(grid))
  measurement_update!(histogram_filter, robot_state, get_states(grid), sensor,
                      range_observation)

  plots=[]
  append!(plots, plot_trajectory(robot_states[1:ii], color=:blue))
  append!(plots, plot_observation(robot_state, range_observation, color=:blue))
  append!(plots, plot_trajectory(target_states[1:ii]))
  append!(plots, visualize_filter(histogram_filter))

  savefig(string("fig/single_robot_gif_", ii, ".png"))

  foreach(x->x.remove(), plots)
end
