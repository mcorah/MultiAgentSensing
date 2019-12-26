using SubmodularMaximization
using POMDPs
using MCTS

grid_size = 10
iterations = 10

grid = Grid(grid_size, grid_size)
sensor = RangingSensor(0.5^2, 0.1^2)

@show robot_state = random_state(grid)

histogram_filter = Filter(grid)

for horizon = 1:2
  problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                             [histogram_filter])

  @show a = solve_single_robot(problem, robot_state,
                               n_iterations = iterations)

  if !in(a, neighbors(grid, robot_state))
    println("Action is not valid")
  end
end
