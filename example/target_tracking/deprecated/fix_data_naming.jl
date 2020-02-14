
if length(ARGS) == 1
  global experiment_name = ARGS[1]
else
  println("Please call with one argument (the experiment name)")
end

println("Processing: ", experiment_name)

isjulia(s) = endswith(s, ".jld2")
is_experiment_file(s) = startswith(s, experiment_name^2) && isjulia(s)

trim_name(s) = s[length(experiment_name)+1:end]

if !isdir(experiment_name)
  println("Making experiment directory: ", experiment_name)
  mkdir(experiment_name)
end

for file in readdir()
  if is_experiment_file(file)
    dest = string(experiment_name, "/", trim_name(file))

    println("Old: ", file, "\nNew: ", dest)

    mv(file, dest)
  end
end
