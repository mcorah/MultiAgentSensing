using SubmodularMaximization
using POMDPs
using MCTS
using D3Trees

grid_size = 10
iterations = 10
horizon = 2

grid = Grid(grid_size, grid_size)
sensor = RangingSensor()

@show robot_state = random_state(grid)

histogram_filter = Filter(grid)

problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                           [histogram_filter])

solver = generate_solver(horizon, n_iterations = iterations,
                         enable_tree_vis=true)
policy = solve(solver, problem)

a, info = action_info(policy, MDPState(robot_state))

D3Tree(info[:tree])
