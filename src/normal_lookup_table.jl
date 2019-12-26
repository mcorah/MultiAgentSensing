using Distributions
using Statistics
using Random

# This is a simple lookup table for the normal pdf
#
# Recall that the normal pdf has the form
#
# 1/(sigma sqrt(2pi)) exp(-1/2 ((x-mu) / sigma)^2)
struct NormalLookup
  # PDF values for 0:increment:max
  #
  # WARNING: since the zvalues start at zero, we have to be careful with zero
  # and one indexing
  pdfvals::Vector{Float64}

  # (unit: standard deviations)
  increment::Float64

  function NormalLookup(;increment, max)
    zvals = 0.0:increment:max

    N = Normal(0.0, 1.0)
    pdfvals = map(x->pdf(N, x), zvals)

    new(pdfvals, increment)
  end
end

function evaluate_no_norm(l::NormalLookup; error::Float64, stddev::Float64)
  if stddev < 0.0
    error("Negative standard deviation: ", stddev)
  end

  # rounding returns the effective zero-index so we add one
  index = round(Int64, abs(error) / (stddev * l.increment)) + 1

  # Since this is a normal pdf, take the limit for out-of-bounds values
  if index > length(l.pdfvals)
    0.0
  else
    l.pdfvals[index]
  end
end
function evaluate(l::NormalLookup; error::Float64, stddev::Float64)
  evaluate_no_norm(l; error=error, stddev=stddev) / stddev
end
