using Base.Iterators
using PyPlot

export Grid, get_states, dims, neighbors, plot_state_space, plot_trajectory,
  random_state, target_dynamics

abstract type StateSpace end
# methods
#
# get_states

struct Grid <: StateSpace
  width::Int64
  height::Int64
end
get_states(g::Grid) = collect(product(1:g.width, 1:g.height))
dims(g::Grid) = (g.width, g.height)

in_bounds(g::Grid, state) = all(x->in(x[1],1:x[2]), zip(state, dims(g)))

function neighbors(g::Grid, state)
  dirs = ((1,0), (0,1))
  offsets = (-1, 0, 1)

  candidates = [state .+ o .* d for d in dirs, o in offsets]
  filter(x->in_bounds(g, x), candidates)
end

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
  scale = 9^2

  i = scatter(xs[1], ys[1], color=color, marker="X", edgecolors=:k, s=scale)
  f = scatter(xs[end], ys[end], color=color, marker="^", edgecolors=:k, s=scale)
  t = plot(xs, ys, color=color)

  # return values are arrays of plot objects, concatenate
  vcat(i, f, t)
end

random_state(g::Grid) = sample(get_states(g))
target_dynamics(g::Grid, s) = sample(neighbors(g, s))
