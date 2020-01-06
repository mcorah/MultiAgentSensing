# Simple test on the weights code

using SubmodularMaximization
using SubmodularMaximization: channel_capacity_by_target,
                              channel_capacity_by_target_time

grid_size = 9
horizon = 5
sensor = RangingSensor(0.5^2, 0.1^2)

# place the target in the center of the grid
target_state = (5,5)

# Some representative and interesting weights sets
state_sets = [[(5,5), (5,5)],
              [(5,0), (5,10)],
              [(5,5), (5,10)],
              [(5,-100), (5, 110)]]

grid = Grid(grid_size, grid_size)
histogram_filter = Filter(grid)

# Create some uncertainty. Two steps is probably a reasonable approximation of
# the kind of target uncertainty that we will encounter
for _ in 1:2
  process_update!(histogram_filter, transition_matrix(grid))
end

for states in state_sets
  @show states

  problem = MultiRobotTargetTrackingProblem(grid, sensor, horizon, [histogram_filter],
                                            states)

  # Investigate weights

  for ii = 1:length(states)
    @show target_weight = channel_capacity_by_target(problem, histogram_filter,
                                                     robot_index=1)

    step_weights = map(1:horizon) do x
      @show channel_capacity_by_target_time(problem, histogram_filter,
                                            robot_index=1, step=x)
    end
  end

  # Compute weight matrices
  methods = [channel_capacities_by_target, channel_capacities_by_target_time]

  for method in methods
    weights = compute_weight_matrix(problem, channel_capacity_method=method)
    total_weight = sum(weights) / 2

    println("Weight: ", total_weight, " (", method, ")")
  end

end

nothing
