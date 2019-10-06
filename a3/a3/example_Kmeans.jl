# Load data
using JLD
X = load("clusterData2.jld","X")

# K-means clustering
k = 4
include("kMeans.jl")
model = kMeans(X,k,doPlot=true)
y = model.predict(X)

include("clustering2Dplot.jl")
clustering2Dplot(X,y,model.W)
