# Plots an random example of a prior and the resulting information distribution

using SubmodularMaximization
using PyPlot
using HistogramFilters

close("all")

steps = 100
grid_size = 10
num_observations = 3

grid = Grid(grid_size, grid_size)
sensor = RangingSensor()

target_state = random_state(grid)

# precomputation
histogram_filter = Filter(grid, target_state)

# Create some fake observations and update the filter
for ii = 1:num_observations
  robot_state = random_state(grid)
  range_observation = generate_observation(grid, sensor, robot_state,
                                           target_state)
  measurement_update!(histogram_filter, robot_state, get_states(grid), sensor,
                      grid, range_observation)
end

@show entropy(histogram_filter)

# Visualize the new prior
visualize_filter(histogram_filter, show_colorbar = false)
title("Prior")


# Information gain on the prior
information = Array(HistogramFilters.get_data(histogram_filter))
for ii = 1:length(information)
  state = SubmodularMaximization.index_to_state(grid, ii)

  information[ii] = finite_horizon_information(grid, histogram_filter, sensor,
                                               [state]).reward
end

# Visualize the information gain
figure()
limits = [0.5, grid_size + 0.5, 0.5, grid_size + 0.5]
image = imshow(information', cmap="viridis", vmin=0.0,
               extent=limits,
               interpolation="nearest", origin="lower")
#make_tight_colorbar(image)
title("Information gain")
