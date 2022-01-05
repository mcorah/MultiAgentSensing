using SubmodularMaximization
using POMDPs
using MCTS
using PyPlot

close()

steps = 100
grid_size = 10
horizon = 5
iterations = 1000

grid = Grid(grid_size, grid_size)
sensor = CoverageSensor()

robot_state = random_state(grid)
target_state = random_state(grid)

target_states = Array{State}(undef, steps)
target_states[1] = target_state

robot_states = Array{State}(undef, steps)
robot_states[1] = robot_state

plot_state_space(grid)
xlim([0, grid_size+1])
ylim([0, grid_size+1])

for ii = 2:steps
  println("Step ", ii)

  # Before the target moves and the robot receives a measurement, execute robot
  # dynamics

  problem = SingleRobotTargetCoverageProblem(grid, sensor, horizon,
                                             [target_state])
  @time solution = solve_single_robot(problem, robot_state,
                                      n_iterations = iterations,
                                      exploration_constant = Float64(horizon))
  global robot_state = solution.action
  trajectory = solution.trajectory
  robot_states[ii] = robot_state

  # Then update the target and sample the observation
  global target_state = target_dynamics(grid, target_states[ii-1])
  target_states[ii] = target_state

  plots=[]
  append!(plots, plot_trajectory(robot_states[1:ii], color=:blue))
  # Replace with a method that plots the range and which targets are in range
  #append!(plots, plot_observation(robot_state, range_observation, color=:blue))
  append!(plots, plot_trajectory(target_states[1:ii]))
  append!(plots, plot_states(trajectory, color=:blue, linestyle=":"))

  coverage = finite_horizon_coverage(grid, target_states[ii-1], sensor,
                                     trajectory)
  println("Reward: ", coverage.reward,
          ", Incremental rewards: ", coverage.incremental_coverage)
  println("Trajectory: ", trajectory)

  line = readline()

  if line == "q"
    break
  end

  if ii < steps
    foreach(x->x.remove(), plots)
  end
end
