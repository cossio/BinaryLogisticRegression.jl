function log_likelihood(w::AbstractVector, X::AbstractMatrix, y::AbstractVector{Bool})
    @assert size(X) == (length(w), length(y))
    inputs = X' * w
    return mean(y .* inputs - log1pexp.(inputs))
end

function log_likelihood_gradient(w::AbstractVector, X::AbstractMatrix, y::AbstractVector{Bool})
    @assert size(X) == (length(w), length(y))
    return X * (y - logistic.(X' * w)) / size(X, 2)
end

function log_likelihood_hessian(w::AbstractVector, X::AbstractMatrix, y::AbstractVector{Bool})
    @assert size(X) == (length(w), length(y))
    inputs = X' * w
    G = logistic.(-inputs) .* logistic.(inputs)
    @tullio H[k,j] := -G[n] * X[k,n] * X[j,n]
    return H / size(X, 2)
end

function logistic_regression_without_bias(X::AbstractMatrix, y::AbstractVector{Bool}; l2::Real = 0, algorithm::Symbol=:newton)
    @assert size(X, 2) == length(y)
    @assert algorithm ∈ (:lbfgs, :newton)
    w = zeros(size(X, 1))

    f(w) = -log_likelihood(w, X, y) + l2 * sum(w.^2) / 2  # objective function
    g(w) = -log_likelihood_gradient(w, X, y) + l2 .* w  # gradient
    h(w) = -log_likelihood_hessian(w, X, y) + l2 * I  # hessian

    w0 = zeros(size(X, 1))  # initial value

    if algorithm === :lbfgs
        sol = optimize(f, g, w0, LBFGS(); inplace = false)
    elseif algorithm === :newton
        sol = optimize(f, g, h, w0, Newton(); inplace = false)
    else
        throw(ArgumentError("Unknown algorithm: $algorithm"))
    end

    return sol.minimizer
end
