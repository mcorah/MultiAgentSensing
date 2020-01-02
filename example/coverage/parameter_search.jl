using PyPlot
using Statistics

using SubmodularMaximization

num_trials = 100
num_agents = 20

f(x) = mean_area_coverage(x, floor(Integer, 20 * num_agents / 4))

struct CoverageParams
  num_sensors
  nominal_area
end

num_range =  collect(2:6)
area_range = collect(0.5:0.1:2.0)

param_set = [CoverageParams(num_sensors, nominal_area) for num_sensors in num_range,
             nominal_area in area_range]

function coverage_performance_ratio(params::CoverageParams)
  println("Testing: sensors ($(params.num_sensors)), nominal area ($(params.nominal_area))")

  sensor_radius = sqrt(params.nominal_area / (num_agents * pi))
  station_radius = 2 * sensor_radius

  agent_specification = CircleAgentSpecification(sensor_radius, station_radius,
                                           params.num_sensors)

  problems = map(1:num_trials) do unused
    agents = generate_agents(agent_specification, num_agents)
    ExplicitPartitionProblem(f, agents)
  end

  myopic_mean = mean(map(x->solve_myopic(x).value, problems))
  sequential_mean = mean(map(x->solve_sequential(x).value, problems))

  myopic_mean / sequential_mean
end

performance_ratios = map(coverage_performance_ratio, param_set)

_, best_index = findmin(performance_ratios)
best_params = param_set[best_index]

println("Best ratio: $(performance_ratios[best_index])")
println("Params: sensors ($(best_params.num_sensors)), nominal area ($(best_params.nominal_area))")

figure()
PyPlot.plt.hist(performance_ratios[:], 10)

figure()
pcolor(performance_ratios, cmap = PyPlot.cm.Blues)
colorbar()

nothing
