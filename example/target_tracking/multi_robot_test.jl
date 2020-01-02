using SubmodularMaximization
using POMDPs
using MCTS

grid_size = 10
iterations = 10
horizon = 5
max_num_robots = 10
num_partitions = 3

grid = Grid(grid_size, grid_size)
sensor = RangingSensor(0.5^2, 0.1^2)


histogram_filter = Filter(grid)

for num_robots = 1:max_num_robots
  println("Running tests for ", num_robots, " robots")

  robot_states = map(x->random_state(grid), 1:num_robots)

  problem = MultiRobotTargetTrackingProblem(grid, sensor, horizon,
                                            [histogram_filter], robot_states)

  println("Solving sequentially")
  solution = solve_sequential(problem)

  println("Solving with ", num_partitions, " partitions")

  println("Serially")
  solve_n_partitions(num_partitions, problem)

  println("Threaded")
  solve_n_partitions(num_partitions, problem, threaded=true)
end
