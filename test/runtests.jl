using Test: @test, @testset, @inferred
using Random: bitrand
using Statistics: mean
using LinearAlgebra: I
using BinaryLogisticRegression: log_likelihood, log_likelihood_gradient, log_likelihood_hessian,
    logistic_regression_without_bias, logistic_regression
using Zygote: gradient, hessian
using LogExpFunctions: logistic

X = randn(7, 100)
y = bitrand(100)
w = randn(7)
l1 = rand()
l2 = rand()

gs = gradient(w) do w
    log_likelihood(w, X, y) - l1 * sum(abs, w) - l2 * sum(w.^2) / 2
end

H = hessian(w) do w
    log_likelihood(w, X, y) - l2 * sum(w.^2) / 2
end

@test @inferred(log_likelihood_gradient(w, X, y) - l1 * sign.(w) - l2 * w) ≈ only(gs)
@test @inferred(log_likelihood_hessian(w, X, y) - l2 * I) ≈ H

@test logistic_regression_without_bias(X, y; algorithm=:lbfgs) ≈ logistic_regression_without_bias(X, y; algorithm=:newton) rtol=1e-4

w, b = logistic_regression(X, y)
@test mean(logistic.(X' * w .+ b)) ≈ mean(y) rtol=1e-8
@test X * y ≈ X * logistic.(X' * w .+ b) rtol=1e-5
