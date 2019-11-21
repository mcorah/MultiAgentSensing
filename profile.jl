using ProfileView
using Statistics

include("SubmodularMaximization.jl")
using .SubmodularMaximization

num_agents = 4
num_sensors = 6
nominal_area = 2.0

sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 2 * sensor_radius

agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                         num_sensors)

agents = generate_agents(agent_specification, num_agents)

f(x) = mean_area_coverage(x, 100)
problem = PartitionProblem(f, agents)

solution = solve_optimal(problem)

times = Float64[]

function foo(n)
  global times = Float64[]

  for i in 1:n
    agents = generate_agents(agent_specification, num_agents)
    problem = PartitionProblem(f, agents)

    time = @elapsed solution = solve_optimal(problem)

    @show solution.value

    push!(times, time)
  end
end

@profview foo(1)
ProfileView.closeall()
@profview foo(50)

println("Mean time: $(mean(times))")

PyPlot.plt.hist(times, 20)

nothing
