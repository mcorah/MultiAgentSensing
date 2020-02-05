# Provides a representative plot demonstrating how target weight varies with
# distance

using SubmodularMaximization
using PyPlot

close()

grid_size = 41
horizon = SubmodularMaximization.default_horizon
sensor = RangingSensor()

# place the target in the center of the grid
target_state = (21,1)

grid = Grid(grid_size, grid_size)

# Create some uncertainty. Two steps is probably a reasonable approximation of
# the kind of target uncertainty that we will encounter
histogram_filter = Filter(grid, target_state)
process_update!(histogram_filter, transition_matrix(grid))

range = 1:grid_size

@time weights = thread_map(range) do x
  state = (x, 1)

  problem = MultiRobotTargetTrackingProblem([state, state],
                                            [histogram_filter],
                                            grid=grid,
                                            horizon=horizon)

  weights = compute_weight_matrix(problem,
                                  channel_capacity_method=
                                    channel_capacities_mcts)

  sum(weights) / 2
end

plot(range, weights)

ratio = maximum(weights)/minimum(weights)
println("Weights falloff ratio: ", ratio)

plot_title = string("Weight Falloff (constant: ", sensor.variance_constant,
                    ", scaling: ", sensor.variance_scaling_factor, ")")
title(plot_title)
xlabel("X position")
ylabel("Weight (bits)")

save_fig("fig", plot_title)

nothing
