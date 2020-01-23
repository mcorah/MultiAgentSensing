# Plots an random example of a prior and the resulting information distribution
#
# This code uses the planner dataset to produce more relevant results

using SubmodularMaximization
using JLD2
using Statistics
using Profile
using ProfileView

println("Loading rollout library")
library_file = "./data/rollout_library.jld2"
@time @load library_file data num_robots trials
rollouts = data

sparse = true

num_information_samples = 10
num_trials = 100

# Overwrite the number of robots
num_robots = 10

if sparse
  global copy_filter(x) = SparseFilter(x; threshold=1e-3)
else
  global copy_filter(x) = Filter(x)
end

# Full problem instances for each trial
trials_data = vcat(map(trials) do trial
                     rollout = rollouts[num_robots, trial]
                     configs = rollout.configs

                     [(grid = configs.grid,
                       sensor = configs.sensor,
                       target_filters = map(copy_filter, sample.target_filters),
                       trajectories = sample.trajectories,
                      )
                      for sample in rollout.samples]
                   end...)

num_filters = length(first(trials_data).target_filters)
samples_per_evaluation = num_filters * num_information_samples

# Create some fake observations and update the filter
function run_test(num_trials; print=true)
  ifprint(xs...) = if print println(xs...) end

  time = @elapsed for trial = trials_data[1:num_trials]
    mean(trial.target_filters) do filter

      finite_horizon_information(trial.grid,
                                 filter,
                                 trial.sensor,
                                 trial.trajectories,
                                 num_samples = num_information_samples
                                ).reward
    end
  end

  ifprint("Test duration: ", time, " seconds")
  ifprint("  ", time / num_trials, " seconds per information estimate")
  ifprint("  ", time / num_trials / samples_per_evaluation,
          " seconds per information estimate per sample")
  ifprint("Note: total number of information samples is ",
          samples_per_evaluation)
end

run_test(1)
@time run_test(num_trials)

## Run the profiler
@profview run_test(1, print=false)

print("Waiting for input...")
readline()

#ProfileView.closeall()
@profview run_test(num_trials, print=false)

println()
Profile.print(mincount=100, format=:flat, sortedby=:count)
