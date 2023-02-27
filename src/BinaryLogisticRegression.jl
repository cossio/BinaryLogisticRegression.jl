module BinaryLogisticRegression

using Statistics: mean
using LinearAlgebra: I
using LogExpFunctions: log1pexp, logistic
using Optim: optimize, LBFGS, Newton
using Tullio: @tullio

include("logistic.jl")
include("logistic_without_bias.jl")

end # module
