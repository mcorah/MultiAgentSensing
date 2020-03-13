# Provides a representative plot demonstrating how target weight varies with
# distance

using SubmodularMaximization
using PyPlot

close("all")

grid_size = 41
horizon = SubmodularMaximization.default_horizon

range_limits = [5.0, 10.0, 15.0, 20.0, Inf]

for range_limit in range_limits
  sensor = RangingSensor(range_limit=range_limit)

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

  label = string("Range. lim.: ", range_limit)
  plot(range, weights, label=label)

  ratio = maximum(weights)/minimum(weights)

  println("Range limit: ", range_limit, " Weights falloff ratio: ", ratio)
end

plot_title = string("Weights Falloff")

title(plot_title)
xlabel("X position")
ylabel("Weight (bits)")
legend()

save_fig("fig", plot_title)

nothing
