function fisher_lda(X::Matrix, y::Vector)
    n, m = size(X)
    classes = unique(y)
    k = length(classes)
    X_classes = [X[y .== c, :] for c in classes]
    mu_classes = [mean(X_classes[i], dims=1)' for i=1:k]
    S_w = [cov(X_classes[i]) for i=1:k]
    S_b = sum([nobs(X_classes[i]) * (mu_classes[i] .- mean(X, dims=1)') * (mu_classes[i] .- mean(X, dims=1)')' for i=1:k])
    _, eigenvecs = eigen(S_b, S_w)
    W = eigenvecs[:, 1:k-1]
    X_lda = X * W
    return X_lda
endo
