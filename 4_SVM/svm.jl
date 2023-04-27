using RDatasets
using Random
using LIBSVM
using MLJ

# "pin" the rng so we get the same results every time.
rng = MersenneTwister(4321)

# Since RDatasets for julia did not offer the Wisconsin Breast Cancer Data Set,
# I'm going with analysis on "High School and Beyond - 1982" Dataset from mlmRev.
# We'll try to predict Sex  given Math score, Socio-economic score, and minority.

# load High School and Beyond - 1982.
hsab = dataset("mlmRev", "Hsb82")

# The dataset is ordered by school. Shuffle to avoid leakage, even though
# we don't use the school in the dataset.
shuffle!(rng, hsab)


#SVM's don't take catagorical data; we should map it to quantitive data.
transform!(hsab, :Sx => ByRow(sex -> sex == "Male" ? 0 : 1) => :Sx)
transform!(hsab, :Minrty => ByRow(minrty -> minrty == "No" ? 0 : 1) => :Minrty)
transform!(hsab, :Sector => ByRow(sector -> sector == "Public" ? 0 : 1) => :Sector)


# 0.2 will be the test set, 0.8 will be the training set.
test, train = partition(hsab, 0.2)

# Feel free to adjust the features to see how the model changes.
testX, testY = unpack(test, (col) -> col ∉ [:Minrty, :School], ==(:Minrty);)
trainX, trainY = unpack(train, (col) -> col ∉ [:Minrty, :School], ==(:Minrty);)

# lambda to "transpose" the matrix, such that the SVM can use it (SVM requires (features x data), dataframe gives (data x features))
shapeData = matrix -> reshape(matrix, (size(testX, 2), :))

testY, trainY = map(df -> convert(AbstractVector, df), [testY, trainY])

# load the SVM model. Another package provides the SVM interface, and we
# load it in to utilize it with MLJ constructs.
svm = svmtrain(shapeData(Matrix(trainX)), trainY)
yhat, decision_values = svmpredict(svm, shapeData(Matrix(testX)))

println(mean(yhat .== testY))