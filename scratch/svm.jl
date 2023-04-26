using LinearAlgebra

data = [
    1.1764 4.2409 0.9750;
    1.0400 3.8676 0.4243;
    1.0979 1.0227 0.4484;
    2.0411 4.7610 0.6668;
    2.0144 4.1217 1.2470;
    2.1454 4.4439 0.3974;
]

labels  = [1; 1; 1; -1; -1; -1;;]
lambdas = [1; 0.7383; 0; 0.411; 1; 0.6872;;]


transformed = (data .* 
    repeat(labels, 1, 3)) .*
    repeat(lambdas, 1, 3)

w = sum(eachrow(transformed))
wb = 3.3149

map(row -> dot(row, w) + 1, eachrow(data))


