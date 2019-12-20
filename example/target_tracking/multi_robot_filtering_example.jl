# Plots target motion and sampled rang observations as circles

using SubmodularMaximization
using PyPlot

close("all")

steps = 100
grid_size = 20
num_robots = 3

grid = Grid(grid_size, grid_size)

robots = map(1:num_robots) do _
  random_state(grid)
end

sensor = RangingSensor(0.5^2, 0.1^2)

initial = random_state(grid)
states = Array{typeof(initial)}(undef, steps)
states[1] = initial

plot_state_space(grid)
xlim([0, grid_size+1])
ylim([0, grid_size+1])

# precomputation
histogram_filter = Filter(grid)

for ii = 2:steps
  println("Step ", ii)
  state = target_dynamics(grid, states[ii-1])
  states[ii] = state

  range_observations = map(robots) do robot
    generate_observation(sensor, robot, state)
  end

  # compute filter updates in place for this script
  process_update!(histogram_filter, transition_matrix(grid))
  for (robot, observation) in zip(robots, range_observations)
    measurement_update!(histogram_filter, robot, get_states(grid), sensor,
                        observation)
  end

  plots=[]
  for (robot, observation) in zip(robots, range_observations)
    append!(plots, plot_robot(robot))
    append!(plots, plot_observation(robot, observation))
  end
  append!(plots, plot_trajectory(states[1:ii]))
  append!(plots, visualize_filter(histogram_filter))

  readline()

  if ii < steps
    foreach(x->x.remove(), plots)
  end
end
