using PyPlot
using Colors

export Circle, AgentSpecification,
  get_block,
  mean_coverage,
  generate_coverage_agents,
  generate_colors,
  mean_area_coverage,
  visualize_agents,
  visualize_solution

# evaluation of the area coverage objective
type Circle
  center::Array{Float64, 1}
  radius::Float64
end

type AgentSpecification
  sensor_radius
  station_radius
  num_sensors
end

type Agent
  center::Array{Float64, 1}
  sensors::Array{Circle, 1}
end

get_block(agent::Agent) = agent.sensors
get_center(agent::Agent) = agent.center


point_in_circle(center, radius, point) = norm(point - center) < radius
point_in_circle(circle::Circle, point) =
  point_in_circle(circle.center, circle.radius, point)

point_covered(circles, point) =
  maximum(map(circle->point_in_circle(circle, point), circles))

mean_coverage(circles, points) =
  mean(map(point->point_covered(circles, point), points))

function mean_area_coverage(circles, discretization)
  num_covered = 0
  for x in linspace(0, 1, discretization), y in linspace(0, 1, discretization)
    num_covered += point_covered(circles, [x; y])
  end

  num_covered / (discretization * discretization)
end


# agent generation
function rand_in_circle()
  r = sqrt(rand())
  theta = 2*pi*rand()
  r * [cos(theta);sin(theta)]
end

function make_agent(agent_specification)
  agent_center = rand(2)

  make_sensor() = Circle(agent_center + agent_specification.station_radius *
                        rand_in_circle(),
                        agent_specification.sensor_radius)

  sensors = [make_sensor() for agent in 1:agent_specification.num_sensors]

  Agent(agent_center, sensors)
end

function generate_coverage_agents(agent_specification, num_agents)
  [make_agent(agent_specification) for agent in 1:num_agents]
end


# visualization
rgb_tuple(color::RGB) = (red(color), green(color), blue(color))

function circle_points(p; radius=1, scale=1, dth=0.1)
  scale*hcat(map(x->p+[radius*cos(x);radius*sin(x)], 0:dth:(2*pi)+dth)...)
end

function plot_circle(p; scale=1, radius=1, color="k", linestyle="-", linewidth=1.0)
  c = circle_points(p, scale=scale, radius=radius)
  plot(c[1,:][:], c[2,:][:], color=color, linestyle=linestyle, linewidth=linewidth)
end

function plot_circle(circle::Circle; x...)
  plot_circle(circle.center; radius = circle.radius, x...)
end

function plot_filled_circle(circle::Circle; color = "k", alpha = 0.3)
  ps = circle_points(circle.center, scale = 1, radius = circle.radius)
  full_circle = [ps ps[:,1]]

  fill(ps[1,:][:], ps[2,:][:], color = color, alpha=alpha, linewidth=0.0)
end

generate_colors(agents::Array{Agent,1}) =
  distinguishable_colors(length(agents) + 1)[2:end]

function visualize_agents(agents::Array{Agent,1}, colors)
  centers = map(get_center, agents);

  map(agents, colors) do agent, color
    center = get_center(agent)

    scatter([center[1]], [center[2]], color = rgb_tuple(color), marker = (5, 2, 0))

    map(circle->plot_circle(circle, color = rgb_tuple(color)), get_block(agent))
  end
  Void
end

function visualize_solution(agents, solution, colors)
  map(agents, colors, solution) do agent, color, selection
    circle = get_block(agent)[selection]

    plot_filled_circle(circle, color = rgb_tuple(color), alpha=0.3)
  end
  Void
end
