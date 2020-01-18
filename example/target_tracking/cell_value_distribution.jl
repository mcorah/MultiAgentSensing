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

# Create a map from number of robots to a vector target filters
filters = Dict(map(num_robots) do num_robots
                 num_robots => vcat(map(trials) do trial
                                      rollout = rollouts[num_robots, trial]

                                      vcat([x.target_filters for x in
                                            rollout.samples]...)
                                    end...)
               end)

# Concatenate and sort filter data (increasing) by robot
probabilities = Dict(map(num_robots) do num_robots
                       my_filters = filters[num_robots]
                       num_robots => sort(vcat(map(my_filters) do filter
                                            @views filter.data[:]
                                          end...))
                     end)

#
# Plot cumulative distribution functions
#

# The cdf value at p is the fraction of the sum of probabilities less than p
xs = 10 .^ (-1 .* reverse(0:.1:10))
map(num_robots) do num_robots
  let probabilities = probabilities[num_robots]

    cdf_vals = map(xs) do p
      index = findfirst(x->x > p, probabilities)

      # If there is no value greater than p, then all is less than p
      total_value = sum(probabilities)
      if isnothing(index)
        1.0
      else
        sum(probabilities[1:index-1]) / total_value
      end
    end

    label = string(num_robots, " robots")
    semilogx(xs, cdf_vals, label=label)
  end
end

ylabel("Cumulative value frac. less than")
xlabel("Filter probability")
legend()
grid()

save_fig("fig", "cell_value_cdf")

#
# Plot the cell value fraction
#

figure()

# We are interested in the fraction of cell values that are less than a given
# value
xs = 10 .^ (-1 .* reverse(0:.1:10))
map(num_robots) do num_robots
  let probabilities = probabilities[num_robots]

    cdf_vals = map(xs) do p
      index = findfirst(x->x > p, probabilities)

      # If there is no value greater than p, then all are less than p
      if isnothing(index)
        1.0
      else
        (index - 1) / length(probabilities)
      end
    end

    label = string(num_robots, " robots")
    semilogx(xs, cdf_vals, label=label)
  end
end

ylabel("Cumulative frac. cells less than")
xlabel("Filter probability")
legend()
grid()

save_fig("fig", "cell_value_fraction")
