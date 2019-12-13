# Show grid and random walk target trajectory

using SubmodularMaximization
using PyPlot

close("all")

steps = 100

grid = Grid(10, 10)


initial = random_state(grid)
states = Array{typeof(initial)}(undef, steps)
states[1] = initial

plot_state_space(grid)

for ii = 2:steps
  println("Step ", ii)
  states[ii] = target_dynamics(grid, states[ii-1])

  plots = plot_trajectory(states[1:ii])

  sleep(0.1)

  if ii < steps
    foreach(x->x.remove(), plots)
  end
end
