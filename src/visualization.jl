using PyCall
using PyPlot
using Colors

matplotlib2tikz = pyimport("matplotlib2tikz")
ag = pyimport("mpl_toolkits.axes_grid1")

###############
# visualization
###############
export to_file, save_fig, make_tight_colorbar, visualize_pdf

to_file(s) = replace(lowercase(s), " " => "_")

function save_fig(fig_path, title)
  matplotlib2tikz.save("$(fig_path)/$(to_file(title)).tex",
                       figureheight="\\figureheight",
                       figurewidth="\\figurewidth",
                       show_info=false)
end

# standard markers
station_center = "+"
sensor_center = (5, 2, 0)
selected_sensor = "*"
event_marker = "x"

agent_scale = 32.0

station_style = "--"


rgb_tuple(color::RGB) = (red(color), green(color), blue(color))

generate_colors(agents) = distinguishable_colors(length(agents) + 1)[2:end]

function visualize_agents(agents, colors)
  centers = map(get_center, agents);

  map(agents, colors) do agent, color
    center = get_center(agent)

#   scatter([center[1]], [center[2]], color = rgb_tuple(color),
#           s = 4*agent_scale, marker = station_center)

    map(get_block(agent)) do element
      plot_element(element; color = rgb_tuple(color))
    end

    plot_circle(Circle(agent.center, agent.radius);
                linestyle=station_style, color = rgb_tuple(color),
                linewidth = 3.0)
  end

  nothing
end

function visualize_solution(p::PartitionProblem, X::ElementArray, agent_colors)
  elements = map(X) do x
    get_element(p.partition_matroid, x)
  end

  colors = [agent_colors[x[1]] for x in X]
  visualize_solution(elements, colors)
end

visualize_solution(p::PartitionProblem, s::Solution, agent_colors) =
  visualize_solution(p, s.elements, agent_colors)

# visualization
function circle_points(p; radius=1, scale=1, dth=0.1)
  scale*hcat(map(x->p+[radius*cos(x);radius*sin(x)], 0:dth:(2*pi)+dth)...)
end

function plot_circle(p; scale=1, radius=1, color="k", linestyle="-", linewidth=1.0, kwargs...)
  c = circle_points(p, scale=scale, radius=radius)
  plot(c[1,:][:], c[2,:][:], color=color, linestyle=linestyle, linewidth=linewidth, kwargs...)
end

function make_tight_colorbar(image)
  ax = gca()

  divider = ag.make_axes_locatable(ax)
  cax = divider.append_axes("right", "5%", pad="3%")
  colorbar(image, cax = cax)

  sca(ax)
end

function visualize_pdf(density::Matrix; limits = [0 1; 0 1], cmap = "viridis",
                       show_colorbar = true)
  ret = []

  xlim = limits[1,:]
  ylim = limits[2,:]

  image = imshow(density', cmap=cmap, vmin=0.0, vmax=maximum(density[:]),
                 extent=[xlim[1], xlim[2], ylim[1], ylim[2]],
                 interpolation="nearest", origin="lower")
  push!(ret, image)

  if show_colorbar
    push!(ret, make_tight_colorbar(image))
  end

  ret
end

function visualize_pdf(fun::Function; n = 1000, limits = [0 1; 0 1], kwargs...)
  xlim = limits[1,:]
  ylim = limits[2,:]

  volume = (xlim[2] - xlim[1]) * (ylim[2] - ylim[1])

  density = [evaluate_pdf(fun, [x,y]) for x in range(xlim[1], stop=xlim[2], length=n),
                                 y in range(ylim[1], stop=ylim[2], length=n)]

  # the density is normalized to be a proper probability density on the input
  # interval so values can be anywhere on real+, keep zero though
  density .= density .* length(density) / sum(density) * volume

  visualize_pdf(density; limits=limits, kwargs...)
end
