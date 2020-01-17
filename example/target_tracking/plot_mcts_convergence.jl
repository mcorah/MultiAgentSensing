# Analyze convergence of Monte Carlo tree search for various horizon lenghts

using SubmodularMaximization
using PyPlot
using Statistics
using Base.Threads
using Base.Iterators
using JLD2
using Printf

close("all")

println("Loading rollout library")
library_file = "./data/rollout_library.jld2"
@time @load library_file data num_robots trials
rollouts = data

# Create a map from number of robots to a vector of configs sample pairs
samples = Dict(map(num_robots) do num_robots
                 num_robots => vcat(map(trials) do trial
                                      rollout = rollouts[num_robots, trial]

                                      [(configs=rollout.configs, sample=sample)
                                       for sample in rollout.samples]
                                    end...)
               end)

# Everything has the same number of samples
sample_inds = 1:length(first(values(samples)))

data_file = "./data/multi_robot_mcts_convergence_data.jld2"
reprocess = false

num_mcts_samples = 2 .^ (4:13)
horizons = 1:6

solver = solve_sequential

all_tests = product(horizons, num_mcts_samples, num_robots, sample_inds)
all_configurations = product(horizons, num_mcts_samples, num_robots)

function get_data()

  # Both guarded by the mutex
  results_mutex = Mutex()
  results = Dict()
  num_tests_completed = 0

  if !isfile(data_file) || reprocess

    iters = collect(all_tests)
    @threads for (horizon, num_mcts_samples, num_robots,
                  sample_ind) in iters

      test = samples[num_robots][sample_ind]

      sample = test.sample

      configs = MultiRobotTargetTrackingConfigs(test.configs;
                                          horizon = horizon,
                                          solver_iterations = num_mcts_samples)


      problem = MultiRobotTargetTrackingProblem(sample.robot_states,
                                                sample.target_filters,
                                                configs)
      solution = solver(problem)
      reward = solution.value

      lock(results_mutex)

      results[horizon, num_mcts_samples, num_robots, sample_ind] = reward
      num_tests_completed +=1

      unlock(results_mutex)

      println("Horizon: ", horizon,
              " Num. samples: ", num_mcts_samples,
              " Num. robots: ", num_robots,
              " Trial: ", sample_ind,
              " (done, ", num_tests_completed, "/", length(all_tests),
              @sprintf(" %0.2f", 100num_tests_completed/length(all_tests)), "%)"
             )
    end

    @save data_file results
  else
    @load data_file results
  end

  results
end

println("Processing rollouts")
results = get_data()

println("Analyzing results")

#
# normalize rewards
#

# Compute the max reward over horizons/num_robots/sample_inds
max_rewards = Dict(map(product(horizons, num_robots, sample_inds)) do x
                     horizon, num_robots, sample_ind = x

                     x => maximum(num_mcts_samples) do mcts_samples
                       results[horizon, mcts_samples, num_robots, sample_ind]
                     end
                   end)

# Normalize rewards over results for different numbers of samples
normalized_rewards = Dict(map(all_tests) do x
                            horizon,num_mcts_samples,num_robots,sample_ind = x
                            configuration = (horizon, num_mcts_samples,
                                             num_robots)

                            max_reward = max_rewards[horizon, num_robots,
                                                     sample_ind]

                            x=>results[x...]/max_reward
                          end)


# Mean reward over all samples for each configuration
mean_rewards = Dict(map(all_configurations) do x
                      x => mean(ind->normalized_rewards[x..., ind], sample_inds)
                    end)

for num_robots in num_robots
  figure()
  for horizon in horizons
    horizon_data = map(num_mcts_samples) do mcts_samples
      mean_rewards[horizon, mcts_samples, num_robots]
    end

    # Print the maximum reward for that horizon and such
    println("Horizon: ", horizon, " Max reward: ", maximum(horizon_data))

    label = string("Horz.: ", horizon)
    loglog(num_mcts_samples, horizon_data, label=label)
  end

  title(string("MCTS convergence (", num_robots, " robots)"))
  xlabel("Num rollouts")
  ylabel("Frac. of max. reward")
  legend(loc="upper left")

  fig_name =  string("mcts_convergence_", num_robots, "_robots")
  save_fig("fig", fig_name)
end

