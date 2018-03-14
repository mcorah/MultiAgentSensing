using SubmodularMaximization
using ProfileView

num_agents = 4
num_sensors = 6
nominal_area = 2.0

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 2 * sensor_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

agents = generate_coverage_agents(agent_specification, num_agents)

f(x) = mean_area_coverage(x, 100)
problem = PartitionProblem(f, agents)

solution = solve_optimal(problem)

times = Float64[]

Profile.clear()
@profile for i in 1:50
  agents = generate_coverage_agents(agent_specification, num_agents)
  problem = PartitionProblem(f, agents)

  time = @elapsed solution = solve_optimal(problem)

  @show solution.value

  push!(times, time)
end

println("Mean time: $(mean(times))")

PyPlot.plt[:hist](times, 20)
ProfileView.view()

Void
