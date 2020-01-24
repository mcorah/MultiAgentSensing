using PyPlot
using LinearAlgebra

export plot_state_space, plot_states, plot_trajectory, plot_robot,
  plot_observation, visualize_filter, visualize_filters

const object_scale = 9^2

function plot_state_space(g::Grid; color=:k)
  states = get_states(g)
  xs = [x[1] for x in states]
  ys = [x[2] for x in states]

  scatter(xs, ys, color=color)
end

# When a robot or target crosses a boundary of the grid, that creates a
# discontinuity.
#
# Cut the trajectory into continuous parts and add segments going off the grid
# to visualize the wrap around.
function wrap_discontinuities(states)
  state_ranges = continuous_ranges(states)
  ranges = convert(Array{Array{Tuple{Float64,Float64}}}, state_ranges)

  # Add segments going off the grid to each discontinuity
  for ii = 1:length(ranges) - 1
    disc_start = last(ranges[ii])
    disc_end = first(ranges[ii+1])

    dir = disc_end .- disc_start
    off_grid = -0.5 .* dir ./ norm(dir)

    push!(ranges[ii], disc_start .+ off_grid)
    insert!(ranges[ii+1], 1, disc_end .- off_grid)
  end

  ranges
end

function plot_states(states::Array{State}; kwargs...)
  vcat(map(wrap_discontinuities(states)) do range
    xs = [x[1] for x in range]
    ys = [x[2] for x in range]

    plot(xs, ys; kwargs...)
  end...)
end

function plot_trajectory(states; color=:red, kwargs...)

  first = states[1]
  last = states[end]

  # start end
  ret = []
  push!(ret, scatter(first[1], first[2], color=color, marker="X", edgecolors=:k,
                     s=object_scale))
  push!(ret, scatter(last[1], last[2], color=color, marker="^", edgecolors=:k,
                     s=object_scale))

  range_plots = map(wrap_discontinuities(states)) do range
    xs = [x[1] for x in range]
    ys = [x[2] for x in range]
    plot(xs, ys; color=color, kwargs...)
  end

  # return values are arrays of plot objects, concatenate
  vcat(ret, range_plots...)
end

function plot_robot(state; color=:blue, kwargs...)
  [scatter(state[1], state[2], color=color, marker="v", edgecolors=:k,
           s=object_scale)]
end

function plot_observation(state, range; linestyle="-", kwargs...)
  plot_circle([state...]; radius=range, linestyle=linestyle, kwargs...)
end

# Visualize a discrete histogram filter
function visualize_filter(filter::AnyFilter; kwargs...)
  visualize_filters([filter]; kwargs...)
end
function visualize_filters(filters::Vector{<:AnyFilter}; show_colorbar = false)
  if length(filters) == 0
    error("No filters to visualize")
  end

  range = get_range(filters[1])

  # Adjust bounds because indices are in centers
  limits = [range[1][[1,end]]'; range[2][[1,end]]'] + repeat([-0.5 0.5], 2)

  data = sum(get_data, filters)

  # Note: visualize filter does not normalize by default
  visualize_pdf(Matrix(data), show_colorbar = show_colorbar, limits=limits)
end
