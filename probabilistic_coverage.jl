using StatsBase

####################
# Sensor definitions
####################

export ProbabilisticAgentSpecification, ProbabilisticSensor

type ProbabilisticAgentSpecification
  max_success_probability::Float64
  sensor_sigma::Float64
  station_radius::Float64
  num_sensors::Float64
end

type ProbabilisticSensor
  center::Array{Float64, 1}
  sensor_sigma::Float64
  max_succsess_probability::Float64
end

function make_agent(agent_specification::ProbabilisticAgentSpecification)
  agent_center = rand(2)

  make_sensor() = ProbabilisticSensor(agent_center
                                    + agent_specification.station_radius *
                                    rand_in_circle(),
                                    agent_specification.sensor_sigma,
                                    agent_specification.max_success_probability)

  sensors = [make_sensor() for agent in 1:agent_specification.num_sensors]

  Agent(agent_center, sensors)
end

##############
# Sensor model
##############

#detection_probability()
#mean_coverage()

###################
# Environment model
###################

export Gaussian, GaussianMixture, sample_pdf, pdf, visualize_pdf, dim,
in_limits, sample_reject

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

###############
# visualization
###############
import Base.convert

convert(::Type{Circle}, s::ProbabilisticSensor) = Circle(s.center, s.sensor_sigma)

function plot_element(sensor::ProbabilisticSensor; x...)
  plot_circle(convert(Circle, sensor); x...)
end

function visualize_solution(sensors::Array{ProbabilisticSensor}, colors)
  map(sensors, colors) do sensor, color
    plot_filled_circle(convert(Circle, sensor), color = rgb_tuple(color),
                       alpha=0.3)
  end
  Void
end
