# Simple test on the weights code

using SubmodularMaximization
using SubmodularMaximization: channel_capacity_by_target,
                              channel_capacity_by_target_time

grid_size = 9
horizon = 5
sensor = RangingSensor()

# place the target in the center of the grid
target_state = (5,5)

# Some representative and interesting weights sets
state_sets = [[(5,5), (5,5)],
              [(5,5), (5,10)],
              [(5,3), (5,7)],
              [(5,1), (5,9)]]

grid = Grid(grid_size, grid_size)
histogram_filter = Filter(grid, target_state)

# Create some uncertainty. Two steps is probably a reasonable approximation of
# the kind of target uncertainty that we will encounter
process_update!(histogram_filter, transition_matrix(grid))

for states in state_sets
  println()

  @show states

  problem = MultiRobotTargetTrackingProblem(grid, sensor, horizon, [histogram_filter],
                                            states)

  # Investigate weights

  for ii = 1:length(states)
    @show target_weight = channel_capacity_by_target(problem, histogram_filter,
                                                     robot_index=ii)

    step_weights = map(1:horizon) do x
      @show channel_capacity_by_target_time(problem, histogram_filter,
                                            robot_index=ii, step=x)
    end
  end

  # Compute weight matrices
  methods = [channel_capacities_mcts,
             channel_capacities_by_target,
             channel_capacities_by_target_time]

  for method in methods
    @time weights = compute_weight_matrix(problem, channel_capacity_method=method)
    total_weight = sum(weights) / 2

    println("Weight: ", total_weight, " (", method, ")")
  end

end

nothing
