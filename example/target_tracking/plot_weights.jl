# Provides a representative plot demonstrating how target weight varies with
# distance

using SubmodularMaximization
using PyPlot

close("all")

grid_size = 41
horizon = SubmodularMaximization.default_horizon

scaling_factors = [0.1, 0.5, 1.0, 2.0, 3.0]

for scaling_factor in scaling_factors
  sensor = RangingSensor(variance_scaling_factor=scaling_factor)

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
                                              sensor=sensor,
                                              horizon=horizon)

    weights = compute_weight_matrix(problem)

    sum(weights) / 2
  end

  label = string("Scal. fact.: ", scaling_factor)
  plot(range, weights, label=label)

  ratio = maximum(weights)/minimum(weights)

  println("Scaling factor: ", scaling_factor, " Weights falloff ratio: ", ratio)
end

plot_title = string("Weights Falloff")

title(plot_title)
xlabel("X position")
ylabel("Weight (bits)")
legend()

save_fig("fig", plot_title)

nothing
