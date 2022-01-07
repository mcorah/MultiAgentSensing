# Save images for trials from large_scale_weights.jl

using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics
using HistogramFilters
using RosDataProcess

close("all")

pygui(false)

experiment_name = "large_scale_weights"
data_folder = "./data"

video_path = "./video"

experiment_file = joinpath(data_folder, experiment_name * ".jld2")

#@load experiment_file results

num_robots = 8:8:100

trials = 1:5
steps = 100
trial_steps = 20:steps

#trial = first(values(results))
#visualize_experiment(trial.trial_data, trial.configs)

for num_robots in num_robots
  println("Saving for ", num_robots, " robots")
  trial = results[num_robots, 1]

  data = trial.trial_data
  configs = trial.configs

  close("all")

  plot_state_space(configs.grid)
  ax = gca()
  ax.axis("off")

  folder = joinpath(video_path, string("num_robots_", num_robots))
  mkpath(folder)

  for (ii, step) in enumerate(data)
    file = joinpath(folder, string(ii, ".jpg"))

    if isfile(file)
      continue
    end

    println("  ", file)

    plots = visualize_time_step(robot_states=step.robot_states,
                                target_states=step.target_states,
                                target_filters=step.target_filters,
                                trajectories=step.trajectories)


    savefig(file, transparent=true, pad_inches=0.01, bbox_inches="tight")

    foreach(x->x.remove(), plots)
  end
end

# Record videos
rate = 2
for num_robots in num_robots
  @show folder = joinpath(video_path, string("num_robots_", num_robots))
  @show file = joinpath(video_path, string("num_robots_", num_robots, ".mp4"))

  video_command = `ffmpeg -hide_banner -y -r $(rate) -i $(folder)/%d.jpg -c:v libx264 -vf "scale=400:-1" $(file)`

  if !isfile(file)
    run(video_command)
  end
end
