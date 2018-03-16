using StatsBase

####################
# Sensor definitions
####################

export ProbabilisticAgentSpecification, ProbabilisticSensor

type ProbabilisticAgentSpecification
  max_success_probability::Float64
  sensor_radius::Float64
  station_radius::Float64
  num_sensors::Float64
end

# detection probability is of the form:
# max_success_probability * exp(-||x-center||^4 / sensor_radius^4)
type ProbabilisticSensor
  center::Array{Float64, 1}
  sensor_radius::Float64
  max_success_probability::Float64
end

function make_agent(agent_specification::ProbabilisticAgentSpecification)
  agent_center = rand(2)

  make_sensor() = ProbabilisticSensor(agent_center
                                    + agent_specification.station_radius *
                                    rand_in_circle(),
                                    agent_specification.sensor_radius,
                                    agent_specification.max_success_probability)

  sensors = [make_sensor() for agent in 1:agent_specification.num_sensors]

  Agent(agent_center, agent_specification.station_radius, sensors)
end

##############
# Sensor model
##############

export detection_probability, mean_detection_probability

function detection_probability(sensor::ProbabilisticSensor, x)
  square_dist = sum((x - sensor.center).^2)

  sensor.max_success_probability * exp(-square_dist^2 / sensor.sensor_radius^4)
end

function detection_probability(sensors::Array{ProbabilisticSensor,1}, x)
  p_fail = reduce(1.0, sensors) do p, sensor
    p * (1.0 - detection_probability(sensor, x))
  end

  1.0 - p_fail
end

function mean_detection_probability(sensors, points)
  sum(map(1:size(points, 2)) do ii
        detection_probability(sensors, points[:,ii])
      end) / length(points)
end

###################
# Environment model
###################

export Gaussian, GaussianMixture, sample_pdf, pdf, visualize_pdf, dim,
in_limits, sample_reject, standard_mixture, generate_events

pdf_coefficient(covariance) = sqrt((2*pi)^size(covariance,1) * det(covariance))

type Gaussian
  mean::Array{Float64,1}
  covariance::Array{Float64,2}
  chol_covariance # result of cholesky decomposition
  inv_covariance # result of inverse
  det_covariance::Float64 # result of determinant
  pdf_coefficient::Float64 # result of pdf_coefficient

  function Gaussian(mean, covariance)
    new(mean, covariance, chol(covariance), inv(covariance), det(covariance),
        pdf_coefficient(covariance))
  end
end

type GaussianMixture
  weights::Array{Float64,1}
  components::Array{Gaussian,1}
end

dim(g::Gaussian) = length(g.mean)

dim(gm::GaussianMixture) = dim(gm.components[1])

function sample_pdf(g::Gaussian)
  g.mean + (randn(1,size(g.covariance,1)) * g.chol_covariance)'
end

function sample_pdf(gm::GaussianMixture)
  normalized_weights = gm.weights / sum(gm.weights)

  sample_pdf(sample(gm.components, Weights(normalized_weights)))
end

in_limits(x, limits) = all(map(1:length(x)) do ii
                             limits[ii,1] <= x[ii] && x[ii] <= limits[ii,2]
                           end)

function sample_reject(p, limits)
  ret = zeros(dim(p), dim(p))
  found_valid = false

  while ~found_valid
    ret = sample_pdf(p)
    found_valid = in_limits(ret, limits)
  end

  ret
end


function pdf{T <: Number}(g::Gaussian, x::Array{T,1})
  d = x - g.mean

  exp(- d' * g.inv_covariance * d / 2)[1] * g.pdf_coefficient
end

function pdf{T <: Number}(gm::GaussianMixture, x::Array{T,1})
  reduce(0.0, zip(gm.weights, gm.components)) do value, w_c
    value + w_c[1] * pdf(w_c[2], x)
  end
end

function visualize_pdf(fun; limits = [0 1; 0 1], n = 1000, cmap = "viridis")
  xlim = limits[1,:]
  ylim = limits[2,:]

  density = [pdf(fun, [x,y]) for x in linspace(xlim[1], xlim[2], n),
                                 y in linspace(ylim[1], ylim[2], n)]

  imshow(density', cmap=cmap, vmin=minimum(density), vmax=maximum(density[:]),
         extent=[xlim[1], xlim[2], ylim[1], ylim[2]],
         interpolation="nearest", origin="lower")
end

function standard_mixture()
  w = [0.30, 0.6, 0.1]
  g1 = Gaussian([0.2, 0.8], diagm([0.004, 0.1]))
  g2 = Gaussian([0.8, 0.2], diagm([0.1, 0.01]))
  g3 = Gaussian([0.7, 0.7], 0.03 * eye(2))

  GaussianMixture(w, [g1, g2, g3])
end

function generate_events(n = 500; limits = [0 1; 0 1])
  dist = standard_mixture()

  hcat(map(x->sample_reject(dist, limits), 1:n)...)
end

###############
# visualization
###############
export visualize_events

function plot_element(circle::ProbabilisticSensor; color = "k")
  center = circle.center
  scatter([center[1]], [center[2]], color = color, s = 0.5 * agent_scale,
          marker = sensor_center, linewidth = 0.5)
end

function visualize_solution(sensors::Array{ProbabilisticSensor}, colors;
                            limits = [0 1; 0 1], n = 1000, cmap = "viridis")
  xlim = limits[1,:]
  ylim = limits[2,:]

  # Highlight the sensors
  map(sensors, colors) do sensor, color
    center = sensor.center
#   scatter([center[1]], [center[2]], color = rgb_tuple(color), s = 2*agent_scale,
#           marker = selected_sensor, edgecolors="k")
  end

  # Plot the actual distribution
  density = [detection_probability(sensors, [x,y])
             for x in linspace(xlim[1], xlim[2], n),
                 y in linspace(ylim[1], ylim[2], n)]

  imshow(density', cmap=cmap, vmin=minimum(density), vmax=maximum(density[:]),
         extent=[xlim[1], xlim[2], ylim[1], ylim[2]],
         interpolation="nearest", origin="lower")

  Void
end

function visualize_events(events)
  scatter(events[1,:]', events[2,:]', color = "k", marker = event_marker)
end
