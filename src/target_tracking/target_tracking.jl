using Base.Iterators

export Grid, get_states, dims, neighbors, random_state, target_dynamics

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

random_state(g::Grid) = sample(get_states(g))
target_dynamics(g::Grid, s) = sample(neighbors(g, s))
