using Printf
using Statistics

# Load variables
include("example_BagOfWords.jl")

# Compute test error with decision tree
for depth = 1:20
	include("decisionTree_infoGain.jl")
	model = decisionTree_infoGain(X,y,depth)
	yhat = model.predict(Xtest)
	testError = mean(yhat .!= ytest)
	@printf("Test error with depth-%d decision tree: %.3f\n",depth,testError)
end
