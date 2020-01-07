using PyPlot

export plot_state_space, plot_states, plot_trajectory, plot_robot,
  plot_observation, visualize_filter, visualize_filters

const object_scale = 9^2

function plot_state_space(g::Grid; color=:k)
  states = get_states(g)
  xs = [x[1] for x in states]
  ys = [x[2] for x in states]

  scatter(xs, ys, color=color)
end

function plot_states(states::Array{State}; kwargs...)
  vcat(map(continuous_ranges(states)) do range
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

  range_plots = map(continuous_ranges(states)) do range
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
function visualize_filter(filter::Filter; kwargs...)
  visualize_filters([filter]; kwargs...)
end
function visualize_filters(filters::Vector{<:Filter}; show_colorbar = false)
  if length(filters) == 0
    error("No filters to visualize")
  end

  range = get_range(filters[1])

  # Adjust bounds because indices are in centers
  limits = [range[1][[1,end]]'; range[2][[1,end]]'] + repeat([-0.5 0.5], 2)

  data = sum(get_data, filters)

  # Note: visualize filter does not normalize by default
  visualize_pdf(data, show_colorbar = show_colorbar, limits=limits)
end
