# Minimal experiment intended to test the experiment runner
# Should plot x^2

using SubmodularMaximization
using PyPlot
using Base.Iterators
using Base.Threads
using JLD2
using Statistics
using Printf
using Random

close("all")

experiment_name = "minimal_experiment"
data_folder = "./data"
reprocess = true

scales = 1:100
repetitions = 10

trial_fun(x) = mean(y->rand(x, x), 1:repetitions)
print_summary(x) = println("Square: ", x)

results = run_experiments(scales,
                          trial_fun=trial_fun,
                          print_summary=print_summary,
                          experiment_name=experiment_name,
                          data_folder=data_folder,
                          reprocess=reprocess
                         )

y = [sum(results[x]) for x in scales]

plot(scales, y)
