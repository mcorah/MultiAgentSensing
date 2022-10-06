using PyPlot
using LinearAlgebra

export plot_state_space, plot_quadrotor, plot_target, plot_states,
  plot_trajectory, plot_robot, plot_observation, visualize_filter,
  visualize_filters

const object_scale = 13^2
const default_linewidth = 2.0
const default_cmap = "viridis"

function plot_state_space(g::Grid; color=:k)
  states = get_states(g)
  xs = [x[1] for x in states]
  ys = [x[2] for x in states]

  scatter(xs, ys, color=color)
end

# Plots a visualization of a quadrotor
rotate(theta) = [cos(theta) -sin(theta); sin(theta) cos(theta)]
function circle(p; radius=1, scale=1, dth=0.1)
  scale*hcat(map(x->p+[radius*cos(x);radius*sin(x)], 0:dth:(2*pi)+dth)...)
end
plot_quadrotor(p::State; kwargs...) = plot_quadrotor([p...]; kwargs...)
function plot_quadrotor(p::Vector; scale=0.20,
                        color="blue",
                        linewidth=default_linewidth,
                        theta=pi/6,
                        alpha=1.0)
  ret = []

  rotor_radius = 0.5

  rot = rotate(theta)

  circle_ps = rot*scale*[[1;0] [0;1] [-1;0] [0;-1]]

  p3 = 0.0
  if length(p) == 3
    p3 = p[3]
  end

  p2 = p[1:2]

  # Plot X shape
  l1 = [p2+scale*rot[:,1] p2-scale*rot[:,1]]
  l2 = [p2+scale*rot[:,2] p2-scale*rot[:,2]]
  append!(ret, plot(l1[1,:][:], l1[2,:][:], p3*ones(2), color=color, linewidth=linewidth,
                  alpha=alpha))
  append!(ret, plot(l2[1,:][:], l2[2,:][:], p3*ones(2), color=color, linewidth=linewidth,
                  alpha = alpha))

  # Plot and fill circles
  for ii = 1:size(circle_ps,2)
    c = circle(p2+circle_ps[:,ii], radius=scale*rotor_radius)
    append!(ret, plot(c[1,:][:], c[2,:][:], p3*ones(size(c,2)), color=color,
                    linestyle="--", linewidth=0.5linewidth, alpha=alpha))
    append!(ret, fill(c[1,:][:], c[2,:][:], p3*ones(size(c,2)), color=:black,
                    alpha=0.3alpha))
  end

  ret
end

function plot_target(p; color="red", kwargs...)
  [scatter(p[1], p[2]; color=color, marker="^", edgecolors=:k,
           s=object_scale, kwargs...)]
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

function plot_states(states::Array{State}; linewidth=default_linewidth,
                     kwargs...)
  vcat(map(wrap_discontinuities(states)) do range
    xs = [x[1] for x in range]
    ys = [x[2] for x in range]

    plot(xs, ys; linewidth=linewidth, kwargs...)
  end...)
end

function plot_trajectory(states; color=:red, linewidth=default_linewidth,
                         kwargs...)

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
    plot(xs, ys; color=color, linewidth=default_linewidth, kwargs...)
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
function visualize_filter(filter::AbstractFilter; kwargs...)
  visualize_filters([filter]; kwargs...)
end
function visualize_filters(filters::Vector{<:AbstractFilter};
                           show_colorbar = false,
                           alpha=0.75)
  if length(filters) == 0
    error("No filters to visualize")
  end

  range = HistogramFilters.get_range(filters[1])

  # Adjust bounds because indices are in centers
  limits = [range[1][[1,end]]'; range[2][[1,end]]'] + repeat([-0.5 0.5], 2)

  data = sum(HistogramFilters.get_data, filters)

  # Note: visualize filter does not normalize by default
  visualize_pdf(Matrix(data), show_colorbar = show_colorbar, limits=limits,
               alpha=alpha, cmap=default_cmap)
end
