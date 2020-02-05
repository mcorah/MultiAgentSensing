# Analyze memory usage for MCTS because it grows to be quite large
#
# This recreates much of the code to plot convergence because that is what first
# caused these issues
#
# Run with: julia --track-allocation=user

using SubmodularMaximization
using Statistics
using Base.Threads
using Base.Iterators
using JLD2
using Profile

num_mcts_samples = 2 ^ 14
horizon = SubmodularMaximization.default_horizon

grid_size = 10
sensor = RangingSensor()

# place the target in the center of the grid
grid = Grid(grid_size, grid_size)
target_state = (5,5)
robot_state = (1,1)

# Create some uncertainty. Two steps is probably a reasonable approximation of
# the kind of target uncertainty that we will encounter
histogram_filter = Filter(grid, target_state)
process_update!(histogram_filter, transition_matrix(grid))

problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                           [histogram_filter])

function run_test(num_mcts_samples)
  # solve via mcts
  trajectory = solve_single_robot(problem, robot_state,
                                  n_iterations = num_mcts_samples).trajectory

  # evaluate mutual information using the default number of samples (which
  # should be quite accurate)
  reward = finite_horizon_information(grid, histogram_filter, sensor,
                                      trajectory).reward
end

run_test(10)

Profile.clear_malloc_data()

println("Running test")
@time run_test(num_mcts_samples)
println("Done")
