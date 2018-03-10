using PyPlot
using SubmodularMaximization

num_agents = 10
num_sensors = 3
nominal_area = 1.0

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 2 * sensor_radius

agent_specification = AgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

agents = generate_coverage_agents(agent_specification, num_agents)

figure()
xlim([0, 1])
ylim([0, 1])

colors = generate_colors(agents)
visualize_agents(agents, colors)
solution = ones(Int64, length(agents))
visualize_solution(agents, solution, colors)

coverage = mean_area_coverage(map((agent, index)->get_block(agent)[index], agents, solution), 1000)

@show coverage
