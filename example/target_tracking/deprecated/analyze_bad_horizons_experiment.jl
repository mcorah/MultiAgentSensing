using SubmodularMaximization
using PyPlot
using JLD2
using Statistics

data_file = "./data/horizon_entropy_data_forgot_to_vary_solvers.jld2"

steps = 100
num_robots = 4

trials = 1:20

horizons = 1:6

# We will drop a fraction of each trial so that the filters have time to
# converge to steady states
drop_fraction = 1/4
trial_steps = (round(Int64, steps * drop_fraction)+1):steps

# Note: we will run trials in threads so the solvers do not have to be threaded
solvers = [solve_sequential, solve_myopic]
solver_strings = map(string, solvers)

@load data_file data

outlier_trial = data[solver_strings[1], 6, 14]
inlier_trial = data[solver_strings[1], 6, 15]

for trial in [outlier_trial, inlier_trial]
  println()
  mean_entropy = mean(x->entropy(x.target_filters), trial.data[trial_steps])
  println("Entropy: ", mean_entropy)
end

plot(map(x->entropy(x.target_filters), outlier_trial.data))
plot(map(x->entropy(x.target_filters), inlier_trial.data))
legend(["outlier", "inlier"])

visualize_experiment(outlier_trial...)
visualize_experiment(inlier_trial...)
