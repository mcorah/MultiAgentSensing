# Analyze convergence of Monte Carlo tree search for various horizon lenghts

using SubmodularMaximization
using PyPlot
using Statistics
using Base.Threads
using Base.Iterators
using JLD2

close()

data_file = "./data/mcvs_convergence_data.jld2"
reprocess = true

num_trials = 100
num_mcts_samples = 2 .^ (4:16)
horizons = 1:6

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

function get_data()
  data = Array{Float64}(undef, num_trials,
                               length(num_mcts_samples),
                               length(horizons))

  if !isfile(data_file) || reprocess
    for (kk, horizon) in Iterators.reverse(enumerate(horizons)),
      (jj, num_mcts_samples) in Iterators.reverse(enumerate(num_mcts_samples))
      println("Horizon: ", horizon, " Num. samples: ", num_mcts_samples)

      problem = SingleRobotTargetTrackingProblem(grid, sensor, horizon,
                                                 [histogram_filter])


      @time for (block_num, block) in enumerate(partition(1:num_trials,
                                                          Threads.nthreads()))
        println("Block num: ", block_num)
        # Ensure that thread execution does not overlap
        @threads for ii in block
          # solve via mcts
          trajectory = solve_single_robot(problem, robot_state,
                                          n_iterations = num_mcts_samples).trajectory

          # evaluate mutual information using the default number of samples (which
          # should be quite accurate)
          reward = finite_horizon_information(grid, histogram_filter, sensor,
                                              trajectory).reward

          data[ii, jj, kk] = reward
        end
      end
    end

    @save data_file data
  else
    global data
    @load data_file data
  end

  data
end

data = get_data()

mean_rewards = reshape(mean(data, dims=1), length.((num_mcts_samples,
                                                    horizons)))

for (ii, horizon) in enumerate(horizons)
  horizon_data = mean_rewards[:,ii]
  maximum_reward = maximum(horizon_data)
  fraction_rewards =  horizon_data ./ maximum_reward

  println("Horizon: ", horizon, " Max reward: ", maximum_reward)

  label = string("Horz.: ", horizon)
  loglog(num_mcts_samples, fraction_rewards, label=label)
end

title("MCTS convergence")
xlabel("Num rollouts")
ylabel("Frac. of max. reward")
legend(loc="upper left")

save_fig("fig", "mcts_convergence")
