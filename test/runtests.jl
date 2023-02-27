using Test: @test, @testset, @inferred
using Random: bitrand
using Statistics: mean
using BinaryLogisticRegression: log_likelihood, log_likelihood_gradient, log_likelihood_hessian,
    logistic_regression_without_bias, logistic_regression
using Zygote: gradient, hessian
using LogExpFunctions: logistic

X = randn(7, 100)
y = bitrand(100)
w = randn(7)

gs = gradient(w) do w
    log_likelihood(w, X, y)
end

H = hessian(w) do w
    log_likelihood(w, X, y)
end

@test @inferred(log_likelihood_gradient(w, X, y)) ≈ only(gs)
@test @inferred(log_likelihood_hessian(w, X, y)) ≈ H

@test logistic_regression_without_bias(X, y; algorithm=:lbfgs) ≈ logistic_regression_without_bias(X, y; algorithm=:newton) rtol=1e-4

w, b = logistic_regression(X, y)
@test mean(logistic.(X' * w .+ b)) ≈ mean(y)
@test X * y ≈ X * logistic.(X' * w .+ b) rtol=1e-5
