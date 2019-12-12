using StatsBase
using LinearAlgebra
using PyCall

ag = pyimport("mpl_toolkits.axes_grid1")

####################
# Sensor definitions
####################

export ProbabilisticAgentSpecification, ProbabilisticSensor

struct ProbabilisticAgentSpecification
  max_success_probability::Float64
  sensor_radius::Float64
  station_radius::Float64
  num_sensors::Float64
end

# detection probability is of the form:
# max_success_probability * exp(-||x-center||^4 / sensor_radius^4)
struct ProbabilisticSensor
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
  p_fail = reduce(sensors; init=1.0) do p, sensor
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

struct Gaussian
  mean::Array{Float64,1}
  covariance::Array{Float64,2}
  chol_covariance # result of cholesky decomposition
  inv_covariance # result of inverse
  det_covariance::Float64 # result of determinant
  pdf_coefficient::Float64 # result of pdf_coefficient

  function Gaussian(mean, covariance)
    new(mean, covariance, cholesky(covariance), inv(covariance), det(covariance),
        pdf_coefficient(covariance))
  end
end

struct GaussianMixture
  weights::Array{Float64,1}
  components::Array{Gaussian,1}
end

dim(g::Gaussian) = length(g.mean)

dim(gm::GaussianMixture) = dim(gm.components[1])

function sample_pdf(g::Gaussian)
  g.mean + (randn(1,size(g.covariance,1)) * g.chol_covariance.U)'
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


function pdf(g::Gaussian, x::Array{T,1}) where T <: Number
  d = x - g.mean

  exp(- d' * g.inv_covariance * d / 2)[1] * g.pdf_coefficient
end

function pdf(gm::GaussianMixture, x::Array{T,1}) where T <: Number
  reduce(zip(gm.weights, gm.components); init=0.0) do value, w_c
    value + w_c[1] * pdf(w_c[2], x)
  end
end

function make_tight_colorbar(image)
  ax = gca()

  divider = ag.make_axes_locatable(ax)
  cax = divider.append_axes("right", "5%", pad="3%")
  colorbar(image, cax = cax)

  sca(ax)
end

function visualize_pdf(fun; limits = [0 1; 0 1], n = 1000, cmap = "viridis")
  xlim = limits[1,:]
  ylim = limits[2,:]

  density = [pdf(fun, [x,y]) for x in range(xlim[1], stop=xlim[2], length=n),
                                 y in range(ylim[1], stop=ylim[2], length=n)]

  # the density is normalized to be a proper probability density on [0, 1]^2
  # so values can be anywhere on real+, keep zero though
  density = length(density) * density / sum(density)
  image = imshow(density', cmap=cmap, vmin=0.0, vmax=maximum(density[:]),
                 extent=[xlim[1], xlim[2], ylim[1], ylim[2]],
                 interpolation="nearest", origin="lower")
  make_tight_colorbar(image)

  nothing
end

function standard_mixture()
  w = [0.30, 0.6, 0.1]
  g1 = Gaussian([0.2, 0.8], Diagonal([0.004, 0.1]))
  g2 = Gaussian([0.8, 0.2], Diagonal([0.1, 0.01]))
  g3 = Gaussian([0.7, 0.7], Diagonal([0.03, 0.03]))

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
  scatter([center[1]], [center[2]], color = color, s = 4*agent_scale,
          marker = sensor_center)
end

function visualize_solution(sensors::Array{ProbabilisticSensor}, colors;
                            limits = [0 1; 0 1], n = 1000, cmap = "viridis")
  xlim = limits[1,:]
  ylim = limits[2,:]

  # Highlight the sensors
  map(sensors, colors) do sensor, color
    center = sensor.center
    scatter([center[1]], [center[2]], color = rgb_tuple(color), s = 4*agent_scale,
            marker = selected_sensor, edgecolors="k")
  end

  # Plot the actual distribution
  probabilities = [detection_probability(sensors, [x,y])
                           for x in range(xlim[1], stop=xlim[2], length=n),
                           y in range(ylim[1], stop=ylim[2], length=n)]

  # detection probability is a direct probability of detection at a given
  # locationand so varies from zero to one
  image = imshow(probabilities', cmap=cmap, vmin=0.0, vmax=1.0,
                 extent=[xlim[1], xlim[2], ylim[1], ylim[2]],
                 interpolation="nearest", origin="lower")
  make_tight_colorbar(image)

  nothing
end

function visualize_events(events)
  scatter(events[1,:]', events[2,:]', color = "k", marker = event_marker,
          s = 2.0 * agent_scale, linewidth = 1.0)
end
