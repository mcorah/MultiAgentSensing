using SubmodularMaximization
using POMDPs
using MCTS

grid_size = 10
horizon = 1
iterations = 10

grid = Grid(grid_size, grid_size)
sensor = RangingSensor(0.5^2, 0.1^2)

@show robot_state = random_state(grid)

histogram_filter = Filter(grid)

solver = generate_solver(horizon, n_iterations = iterations)

mdp = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                       [histogram_filter])
policy = solve(solver, mdp)

@show a = action(policy, MDPState(robot_state))

if !in(a, neighbors(grid, robot_state))
  println("Action is not valid")
end
