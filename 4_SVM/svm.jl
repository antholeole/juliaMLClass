using RDatasets
using Random
using LIBSVM
using MLJ

# "pin" the rng so we get the same results every time.
rng = MersenneTwister(1234)

# Since RDatasets for julia did not offer the Wisconsin Breast Cancer Data Set,
# I'm going with analysis on "High School and Beyond - 1982" Dataset from mlmRev.
# We'll try to predict MAch (Math Achivement) given Sex, Socio-economic score,
# and minority.

# load High School and Beyond - 1982.
hsab = dataset("mlmRev", "Hsb82")

# The dataset is ordered by school. Shuffle to avoid leakage, even though
# we don't use the school in the dataset.
shuffle!(rng, hsab)

#SVM's don't take catagorical data; we should map it to quantitive data.
transform!(hsab, :Sx => ByRow(sex -> sex == "Male" ? 0 : 1) => :Sx)
transform!(hsab, :Minrty => ByRow(minrty -> minrty == "No" ? 0 : 1) => :Minrty)

# 0.2 will be the test set, 0.8 will be the training set
test, train = partition(hsab, 0.2) 
testX, testY = unpack(test, in([:Sx, :SSS, :Minrty]), ==(:MAch);)
trainX, trainY = unpack(train, in([:Sx, :SSS, :Minrty]), ==(:MAch);)

testY, trainY = map(df -> convert(AbstractVector, df), [testY, trainY])

# load the SVM model. Another package provides the SVM interface, and we
# load it in to utilize it with MLJ constructs.
svm = svmtrain(reshape(Matrix(trainX), (3, :)), trainY)
yhat, decision_values = svmpredict(svm, reshape(Matrix(trainX), (3, :))

println(yhat[1:100])
println(mean(yhat .== testY) * 100)