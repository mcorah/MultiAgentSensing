using SubmodularMaximization
using PyPlot
using Statistics
using Base.Threads

close()

grid_size = 10
horizon = 5
sensor = RangingSensor()

# place the target in the center of the grid
target_state = (5,5)
robot_state = (1,1)

grid = Grid(grid_size, grid_size)

# Create some uncertainty. Two steps is probably a reasonable approximation of
# the kind of target uncertainty that we will encounter
histogram_filter = Filter(grid, target_state)
process_update!(histogram_filter, transition_matrix(grid))

problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                           [histogram_filter])

num_trials = 100
num_mcts_samples = 2 .^ (4:20)

mean_rewards = map(num_mcts_samples) do num_mcts_samples
  println("Solving at ", num_mcts_samples)

  # Run tests in threads
  results = zeros(num_trials)
  @time for ii = 1:num_trials
    println("  Trial: ", ii)
    # solve via mcts
    trajectory = solve_single_robot(problem, robot_state,
                                    n_iterations = num_mcts_samples).trajectory

    # evaluate mutual information using the default number of samples (which
    # should be quite accurate)
    reward = finite_horizon_information(grid, histogram_filter, sensor,
                                        trajectory).reward

    results[ii] = reward
  end

  mean(results)
end

@show maximum_reward = maximum(mean_rewards)

fraction_rewards =  mean_rewards ./ maximum_reward

loglog(num_mcts_samples, fraction_rewards)
title("MCTS convergence")
xlabel("Num rollouts")
ylabel("Frac. of max. reward")

save_fig("fig", "mcts_convergence")
