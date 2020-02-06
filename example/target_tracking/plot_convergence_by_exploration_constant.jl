# Analyze convergence of Monte Carlo tree search while varying the exploration
# constant

using SubmodularMaximization
using PyPlot
using Statistics
using Base.Threads
using Base.Iterators
using JLD2

close()

data_folder = "./data"
experiment_name = "mcts_exploration_constant_data"
reprocess = false

num_trials = 1:100
num_mcts_samples = map(x->round(Int64, x), 1.3 .^ (15:26))

horizon = SubmodularMaximization.default_horizon

exploration_constants =  1.03 .* [0.1, 0.5, 0.7, 1.0, 1.3, 1.5, 10.0]

grid_size = 12
const sensor = RangingSensor()

# place the target in the center of the grid
const grid = Grid(grid_size, grid_size)
const target_state = (8,8)
const robot_state = (3,3)

# Create some uncertainty. Two steps is probably a reasonable approximation of
# the kind of target uncertainty that we will encounter
const histogram_filter = Filter(grid, target_state)
process_update!(histogram_filter, transition_matrix(grid))

const problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                                 [histogram_filter])

all_tests = product(exploration_constants, num_mcts_samples, num_trials)
all_configurations = product(exploration_constants, num_mcts_samples)

function trial_fun(x)
  (exploration_constant, num_mcts_samples, trial) = x

  trajectory = solve_single_robot(problem, robot_state,
                                  n_iterations=num_mcts_samples,
                                  exploration_constant=exploration_constant
                                 ).trajectory

  # evaluate mutual information using the default number of samples (which
  # should be quite accurate)
  finite_horizon_information(grid, histogram_filter, sensor,
                             trajectory).reward
end
function print_summary(x)
  (exploration_constant, num_mcts_samples, trial) = x

  println("Exploration constant: ", exploration_constant,
          " Num. samples: ", num_mcts_samples,
          " Trial: ", trial,
         )
end

@time data = run_experiments(all_tests,
                             trial_fun=trial_fun,
                             print_summary=print_summary,
                             experiment_name=experiment_name,
                             data_folder=data_folder,
                             reprocess=reprocess,
                            )

mean_rewards = map(all_configurations) do configuration
  mean(x->data[configuration..., x], num_trials)
end

maximum_reward = maximum(mean_rewards)

for (ii, exploration_constant) in enumerate(exploration_constants)
  data = mean_rewards[ii,:]
  fraction_rewards =  data ./ maximum_reward

  label = string("Exp. const.: ", exploration_constant)
  loglog(num_mcts_samples, fraction_rewards, label=label)
end

title("MCTS convergence")
xlabel("Num rollouts")
ylabel("Frac. of max. reward")
legend(loc="upper left")

save_fig("fig", "mcts_exploration_constants")
