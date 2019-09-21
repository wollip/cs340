using Printf
using Statistics
using Random
include("misc.jl")
include("clustering2Dplot.jl")

mutable struct PartitionModel
	predict # Function for clustering new points
	y # Cluster assignments
	W # Prototype points
end

function kMeans(X,k;doPlot=false)
# K-means clustering

(n,d) = size(X)

# Choos random points to initialize means
W = zeros(k,d)
perm = randperm(n)
for c = 1:k
	W[c,:] = X[perm[c],:]
end

# Initialize cluster assignment vector
y = zeros(Int64, n)
changes = n
while changes != 0

	# Compute (squared) Euclidean distance between each point and each mean
	D = distancesSquared(X,W)

	# Degenerate clusters will distance NaN, change to Inf
	# (since Julia thinks NaN is smaller than all other numbers)
	D[findall(isnan.(D))] .= Inf

	# Assign each data point to closest mean (track number of changes labels)
	changes = 0
	for i in 1:n
		(~,y_new) = findmin(D[i,:])
		changes += (y_new != y[i])
		y[i] = y_new
	end

	# Optionally visualize the algorithm steps
	if doPlot && d == 2
		clustering2Dplot(X,y,W)
		sleep(.1)
	end

	# Find mean of each cluster
	for c in 1:k
		W[c,:] = mean(X[y.==c,:],dims=1)
	end

	# Optionally visualize the algorithm steps
	if doPlot && d == 2
		clustering2Dplot(X,y,W)
		sleep(.1)
	end

	@printf("Running k-means, changes = %d\n",changes)
end

function predict(Xhat)
	(t,d) = size(Xhat)

	D = distancesSquared(Xhat,W)

	yhat = zeros(Int64,t)
	for i in 1:t
		(~,yhat[i]) = findmin(D[i,:])
	end
	return yhat
end

return PartitionModel(predict,y,W)
end

function kMeansError(X,y,W)
	(n,d) = size(X)

	f = 0
	for i in 1:n
		for j = 1:d
			f += (X[i,j] - W[y[i],j])^2
		end
	end
	return f
end

kMeansError2(X,y,W) = sum((X - W[y,:]).^2)

