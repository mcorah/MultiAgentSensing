using FileIO
using SubmodularMaximization
using JLD2
using Printf

experiment_name = "weights_by_number_of_robots"
cache_folder = string("./data/", experiment_name, "/")

file_pattern = r".*jld2"

for file in readdir(cache_folder)
  if occursin(file_pattern, file)
    print("Reendocding: ", file, "...")

    file_path = string(cache_folder, file)
    data = load(file_path)

    trial_weights, trial_data, configs = first(values(data))

    @save file_path trial_weights trial_data configs

    println("done")
  end
end
