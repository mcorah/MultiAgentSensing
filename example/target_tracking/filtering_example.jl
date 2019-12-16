# Plots target motion and sampled rang observations as circles

using SubmodularMaximization
using PyPlot

close("all")

steps = 100
grid_size = 10

grid = Grid(grid_size, grid_size)

robot_state = div.((grid_size, grid_size), 2)
sensor = RangingSensor(0.5^2, 0.1^2)

initial = random_state(grid)
states = Array{typeof(initial)}(undef, steps)
states[1] = initial

plot_state_space(grid)
xlim([0, grid_size+1])
ylim([0, grid_size+1])

# precomputation
grid_states = get_states(grid)
transition = transition_matrix(grid, states = grid_states)
histogram_filter = Filter(grid)

for ii = 2:steps
  println("Step ", ii)
  state = target_dynamics(grid, states[ii-1])
  states[ii] = state

  range_observation = generate_observation(sensor, robot_state, state)

  # compute filter updates in place for this script
  process_update!(histogram_filter, transition)
  measurement_update!(histogram_filter, robot_state, grid_states, sensor,
                      range_observation)

  plots=[]
  append!(plots, plot_robot(robot_state))
  append!(plots, plot_observation(robot_state, range_observation))
  append!(plots, plot_trajectory(states[1:ii]))
  append!(plots, visualize_filter(histogram_filter))

  readline()

  if ii < steps
    foreach(x->x.remove(), plots)
  end
end
