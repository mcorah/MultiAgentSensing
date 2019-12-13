using PyPlot

export plot_state_space, plot_trajectory, plot_robot, plot_observation

const object_scale = 9^2

function plot_state_space(g::Grid; color=:k)
  states = get_states(g)
  xs = [x[1] for x in states]
  ys = [x[2] for x in states]

  scatter(xs, ys, color=color)
end

function plot_trajectory(states; color=:red)

  xs = [x[1] for x in states]
  ys = [x[2] for x in states]

  # start end
  i = scatter(xs[1], ys[1], color=color, marker="X", edgecolors=:k,
              s=object_scale)
  f = scatter(xs[end], ys[end], color=color, marker="^", edgecolors=:k,
              s=object_scale)
  t = plot(xs, ys, color=color)

  # return values are arrays of plot objects, concatenate
  vcat(i, f, t)
end

function plot_robot(state; color=:blue, kwargs...)
  [scatter(state[1], state[2], color=color, marker="v", edgecolors=:k,
           s=object_scale)]
end

function plot_observation(state, range; linestyle="-", kwargs...)
  plot_circle([state...]; radius=range, linestyle=linestyle, kwargs...)
end
