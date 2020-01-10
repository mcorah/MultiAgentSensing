using Printf
using Coverage

const default_pid = 52343

function analyze_dir(dir; pid=default_pid)
  pid_string = string(pid)
  filter(x->occursin(pid_string, x.filename), Coverage.analyze_malloc(dir))
end

# Projects containing code
dirs = [homedir() * "/projects/distributed_sensor_coverage",
        homedir() * "/projects/center_of_mass",
        homedir() * "/.julia/dev/MCTS"]

info = vcat(map(analyze_dir, dirs)...)

# sort by decreasing number of bytes
sort!(info, by=x->x.bytes, rev=true)

for line in info[1:30]
  @printf("%12d", line.bytes)
  println("  ", line.filename, "(", line.linenumber, ")")
end
