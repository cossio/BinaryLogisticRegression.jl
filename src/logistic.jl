
function logistic_regression(X::AbstractMatrix, y::AbstractVector{Bool}; l2::Real = 0)
    @assert size(X, 2) == length(y)
    _w = logistic_regression_without_bias([X ; ones(eltype(X), size(X, 2))], y; l2)
    w = _w[1:end - 1]
    b = _w[end]
    return w, b
end
