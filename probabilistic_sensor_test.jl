using PyPlot

using SubmodularMaximization

discretization = 1000

num_agents = 20
num_sensors = 6
nominal_area = 1.0

max_success_probability = 0.9
sensor_radius = sqrt(nominal_area / (num_agents * pi))
station_radius = 10 * sensor_radius

sensor = ProbabilisticSensor([0.5, 0.5], sensor_radius, max_success_probability)

density = [detection_probability(sensor, [x,y])
           for x in range(0, stop=1, length=discretization),
           y in range(0, stop=1, length=discretization)]

figure()
imshow(density', cmap="viridis", vmin=minimum(density), vmax=maximum(density[:]),
       extent=[0, 1, 0, 1], interpolation="nearest", origin="lower")
colorbar()

nothing
