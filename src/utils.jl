# Some general utilities

export thread_map

function thread_map(f::Function, x)
  buffer = Array{Any}(undef, size(x))

  input = collect(x)

  Threads.@threads for ii in 1:length(x)
    buffer[ii] = f(input[ii])
  end

  # Infer the return type using the union of element types
  Array{Union{typeof.(buffer)...}}(buffer)
end

function thread_map(f::Function, x, output_type::Type{T}) where T
  buffer = Array{T}(undef, size(x))

  input = collect(x)

  Threads.@threads for ii in 1:length(x)
    buffer[ii] = f(input[ii])
  end

  buffer
end
