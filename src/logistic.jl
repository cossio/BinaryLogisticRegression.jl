
function logistic_regression(X::AbstractMatrix, y::AbstractVector{Bool}; l2::Real = 0, algorithm::Symbol=:newton)
    @assert size(X, 2) == length(y)
    _X = [X ; ones(eltype(X), 1, size(X, 2))]
    _w = logistic_regression_without_bias(_X, y; l2, algorithm)
    w = _w[1:end - 1]
    b = _w[end]
    return w, b
end
