using RDatasets

# Since RDatasets for julia did not offer the Wisconsin Breast Cancer Data Set,
# I'm going with analysis on "High School and Beyond - 1982" Dataset from mlmRev.
# We'll try to predict MAch (Math Achivement) given Sex, Socio-economic score,
# and minority.

# load High School and Beyond - 1982
cancer = dataset("mlmRev", "Hsb82")

X = Matrix(select!(cancer, [:Sx, :SSS, :Minrty]))
