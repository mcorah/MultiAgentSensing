# Some general utilities

using Base.Threads

export thread_map, spawn_for_each

function thread_map(f::Function, x)
  buffer = Array{Any}(undef, size(x))

  input = collect(x)

  @threads for ii in 1:length(x)
    buffer[ii] = f(input[ii])
  end

  # Infer the return type using the union of element types
  Array{Union{typeof.(buffer)...}}(buffer)
end

function thread_map(f::Function, x, output_type::Type{T}) where T
  buffer = Array{T}(undef, size(x))

  input = collect(x)

  @threads for ii in 1:length(x)
    buffer[ii] = f(input[ii])
  end

  buffer
end

# Runs a simple worker on a number of tasks
# Returns *nothing*
function spawn_for_each(f::Function, X)
  next_job = Atomic{Int64}(1)

  # Spawn simple workers to run the tasks
  @threads for _ in 1:nthreads()
    while (index = atomic_add!(next_job, 1)) <= length(X)
      f(X[index])
    end
  end

  nothing
end
