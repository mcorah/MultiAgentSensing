using SubmodularMaximization
using POMDPs
using MCTS
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2

close()

data_file = "./data/horizon_entropy_data.jld2"
reprocess = false

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

function get_data()
  data_mutex = Mutex()
  data = Dict()

  if !isfile(data_file) || reprocess
    tests = product(solvers, horizons, trials)
    for (ii, block) in enumerate(partition(tests, nthreads()))
      println("Block num: ", ii)

      @time @threads for (solver, horizon, trial) in block

        configs = MultiRobotTargetTrackingConfigs(num_robots,
                                                  horizon=horizon,
                                                  solver_iterations=10,
                                                 )

        trial_data = target_tracking_experiment(steps=steps, num_robots=num_robots,
                                                configs=configs)

        lock(data_mutex)
        data[string(solver), horizon, trial] =
          (data=trial_data, configs=configs)
        unlock(data_mutex)
      end
    end

    @save data_file data
  else
    @load data_file data
  end

  data
end

@time data = get_data()

entropies = Dict(key=>map(x->sum(entropy, x.target_filters), value.data)
                 for (key, value) in data)

for solver in solver_strings
  figure()

  data = map(horizons) do horizon
    vcat(map(trials) do trial
      entropies[solver, horizon, trial][trial_steps]
    end...)
  end

  boxplot(data, notch=false, vert=false)
  title(string(solver))
  xlabel("Entropy (bits)")
end
