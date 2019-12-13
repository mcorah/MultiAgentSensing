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

for ii = 2:steps
  println("Step ", ii)
  state = target_dynamics(grid, states[ii-1])
  states[ii] = state

  range_observation = generate_observation(sensor, robot_state, state)

  plots=[]
  append!(plots, plot_robot(robot_state))
  append!(plots, plot_observation(robot_state, range_observation))
  append!(plots, plot_trajectory(states[1:ii]))

  sleep(0.1)

  if ii < steps
    foreach(x->x.remove(), plots)
  end
end
