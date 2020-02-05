# The purpose of this script is to bootstrap a library of planning subproblems
# that are representative of (what should be) a high performance planner

using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics

close()

data_file = "./data/rollout_library.jld2"
reprocess = false

steps = 60
num_robots = [4, 6, 8, 10]

trials = 1:20

horizon = SubmodularMaximization.default_horizon

# Keep samples from only a few (8) relevant time-steps in every trial
sample_steps = 20:5:steps

# At this time, the appropriate number of samples for convergence is unknown
# Using a relatively large number can improve results
num_iterations = 3000

solver = solve_sequential

function get_data()
  data_mutex = Mutex()
  data = Dict()

  global num_robots, trials

  if !isfile(data_file) || reprocess

    println("Running trials")

    tests = collect(product(num_robots, trials))
    @threads for (num_robots, trial) in tests

      configs = MultiRobotTargetTrackingConfigs(num_robots,
                                                horizon=horizon,
                                                solver_iterations=
                                                num_iterations
                                               )

      @time rollout = target_tracking_experiment(steps=steps,
                                           num_robots=num_robots,
                                           configs=configs,
                                           solver=solver
                                          )

      # Samples correspond to joint system states where I will wish to compare
      # solvers
      samples = rollout[sample_steps]

      lock(data_mutex)
      data[num_robots, trial] =
      (rollout=rollout, samples=samples, configs=configs)
      unlock(data_mutex)

      println("Robots ", num_robots, " Trial ", trial, " (done)")
    end

    @save data_file data num_robots trials
  else
    @load data_file data num_robots trials
  end

  data
end

@time data = get_data()

entropies = Dict(key=>map(x->entropy(x.target_filters), value.samples)
                 for (key, value) in data)

concatenated_entropy = map(num_robots) do num_robots
  vcat(map(trials) do trial
    entropies[num_robots, trial]
  end...)
end[:]

objective_values = map(num_robots) do num_robots
  @show num_robots
  vcat(map(trials) do trial
    map(x->x.objective, data[num_robots, trial].samples)
  end...)
end[:]

for (ii, num_robots) in enumerate(num_robots)
  mean_objective = mean(objective_values[ii])
  mean_entropy = mean(concatenated_entropy[ii])

  println("Robots (", num_robots, ")")
  println("  Mean entropy: ", mean_entropy,
          " (", mean_entropy / num_robots, " per robot)")
  println("  Mean objective: ", mean_objective,
          " (", mean_objective / num_robots, " per robot)")
end
