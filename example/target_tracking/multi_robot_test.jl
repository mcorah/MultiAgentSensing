using SubmodularMaximization
using POMDPs
using MCTS

grid_size = 10
iterations = 10
horizon = 5
max_num_robots = 3

grid = Grid(grid_size, grid_size)
sensor = RangingSensor(0.5^2, 0.1^2)


histogram_filter = Filter(grid)

for num_robots = 1:max_num_robots

  robot_states = map(x->random_state(grid), 1:num_robots)

  problem = MultiRobotTargetTrackingProblem(grid, sensor, horizon,
                                            [histogram_filter], robot_states)

  @show solution = solve_sequential(problem)
end
