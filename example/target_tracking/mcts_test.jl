using SubmodularMaximization
using POMDPs
using MCTS

grid_size = 10
iterations = 10

grid = Grid(grid_size, grid_size)
sensor = RangingSensor()

@show robot_state = random_state(grid)

histogram_filter = Filter(grid)

for horizon = 1:2
  problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                             [histogram_filter])

  solution = solve_single_robot(problem, robot_state,
                                n_iterations = iterations)

  @show solution.action
  @show solution.trajectory

  if !in(solution.action, neighbors(grid, robot_state))
    println("Action is not valid")
  end
end
