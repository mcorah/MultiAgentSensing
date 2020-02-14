using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics
using Printf
using Random
using Histograms

close("all")

experiment_name = "entropy_by_solver"
data_folder = "./data"

reprocess = false

steps = 100
num_robots = 8:8:48

trials = 1:20

# We will drop a fraction of each trial so that the filters have time to
# converge to steady states
trial_steps = 20:steps

# Note: we will run trials in threads so the solvers do not have to be threaded
horizon = SubmodularMaximization.default_horizon

solver_rounds = [2, 4, 8]
solvers = [solve_random,
           solve_myopic,
           map(num->prob->solve_n_partitions(num, prob), solver_rounds)...,
           solve_sequential]
solver_strings = ["random",
                  "myopic",
                  map(x->string("dist. ",x," rnds."), solver_rounds)...,
                  "sequential"]
solver_inds = 1:length(solvers)

all_tests = product(solver_inds, num_robots, trials)
all_configurations = product(solver_inds, num_robots)


function trial_fun(x)
  (solver_ind, num_robots, trial) = x

  configs = MultiRobotTargetTrackingConfigs(num_robots, horizon=horizon)

  trial_data = target_tracking_experiment(steps=steps,
                                          num_robots=num_robots,
                                          configs=configs,
                                          solver=solvers[solver_ind]
                                         )
  (data=trial_data, configs=configs)
end
function print_summary(x)
  (solver_ind, num_robots, trial) = x

  println("Num. Robots: ", num_robots,
          " Solver: ", solver_strings[solver_ind],
          " Trial: ", trial)
end

@time results = run_experiments(all_tests,
                             trial_fun=trial_fun,
                             print_summary=print_summary,
                             experiment_name=experiment_name,
                             data_folder=data_folder,
                             reprocess=reprocess
                            )

# Normalize by number of targets
entropies = Dict(map(all_tests) do key
                   key => map(results[key].data) do x
                     entropy(x.target_filters) / length(x.target_filters)
                   end
                 end)

# Concatenate all data for a set of trials
concatenated_entropy = Dict(map(all_configurations) do configuration
  trial_data = map(trials) do trial
    entropies[configuration..., trial][trial_steps]
  end

  configuration=>vcat(trial_data...)
end[:])

titles = map(all_configurations) do (solver_ind, num_robots)
  string(num_robots, "-robots ", solver_strings[solver_ind])
end[:]

boxplot([concatenated_entropy[c] for c in all_configurations][:],
        notch=false,
        vert=false)

title("Entropy distribution")
xlabel("Entropy per target (bits)")
yticks(1:length(titles), titles)

save_fig("fig", "entropy_by_solver")

# Plot means to get output in an easier format

figure()

# Plot everything in a given order to get the labels to line up nicely
solver_order = [1, 2, 6, 3, 4, 5]

plot_args = Dict(3=>Dict(:color=>:k),
                 4=>Dict(:color=>:k, :linestyle=>"--"),
                 5=>Dict(:color=>:k, :linestyle=>"-."))

for solver_ind in solver_order
  entropies = map(n->concatenated_entropy[solver_ind, n], num_robots)

  means = map(mean, entropies)
  label = solver_strings[solver_ind]

  # Process args
  kwargs = get(plot_args, solver_ind, Dict())

  plot(num_robots, means, label=label; kwargs...)
end

ylabel("Entropy per target (bits)")
xlabel("Num. robots")
legend(loc="upper right", ncol=3)
grid()

save_fig("fig", "entropy_by_solver_plot")
