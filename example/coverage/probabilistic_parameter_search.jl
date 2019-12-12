using PyPlot
using Statistics

using SubmodularMaximization

num_trials = 50

num_events = 50
num_agents = 50
num_sensors = 10
max_success_probability = 1.0

struct ProbabilisticCoverageParams
  nominal_area
  radius_factor
end

area_range = collect(0.2:0.4:3.0)
radius_factor_range = collect(1:1.0:8.0)

param_set = [ProbabilisticCoverageParams(nominal_area, radius_factor)
             for nominal_area in area_range,
             radius_factor in radius_factor_range]

function performance_ratio(params::ProbabilisticCoverageParams)
  println("Testing: area ($(params.nominal_area)), radius-factor ($(params.radius_factor))")

  sensor_radius = sqrt(params.nominal_area / (num_agents * pi))
  station_radius = params.radius_factor * sensor_radius

  agent_specification = ProbabilisticAgentSpecification(max_success_probability,
                                                        sensor_radius,
                                                        station_radius,
                                                        num_sensors)

  problems = map(1:num_trials) do unused
    events = generate_events(num_events)
    f(x) = mean_detection_probability(x, events)

    agents = generate_agents(agent_specification, num_agents)

    PartitionProblem(f, agents)
  end

  myopic_mean = mean(map(x->solve_myopic(x).value, problems))
  sequential_mean = mean(map(x->solve_sequential(x).value, problems))

  myopic_mean / sequential_mean
end

performance_ratios = map(performance_ratio, param_set)

_, best_index = findmin(performance_ratios)
best_params = param_set[best_index]

println("Best ratio: $(performance_ratios[best_index])")
println("Params: area ($(best_params.nominal_area)), radius-factor ($(best_params.radius_factor))")

figure()
PyPlot.plt.hist(performance_ratios[:], 10)

figure()
pcolormesh(area_range, radius_factor_range, performance_ratios', cmap="Blues")
colorbar()
xlabel("nominal area")
ylabel("radius factor")

nothing
