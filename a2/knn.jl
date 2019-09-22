include("misc.jl") # Includes GenericModel typedef

function knn_predict(Xhat,X,y,k)
  (n,d) = size(X)
  (t,d) = size(Xhat)
  k = min(n,k) # To save you some debuggin
    
  distancesMatrix = distancesSquared(Xhat, X)
  yhat = zeros(t)
  for i in 1:t
    perm = partialsortperm(distancesMatrix[i, :], 1:k)
    yhat[i] = mode(y[perm])
  end
  return yhat
end

function knn(X,y,k)
	# Implementation of k-nearest neighbour classifier
  predict(Xhat) = knn_predict(Xhat,X,y,k)
  return GenericModel(predict)
end

function cknn(X,y,k)
	# Implementation of condensed k-nearest neighbour classifier
	(n,d) = size(X)
	Xcond = X[1,:]'
	ycond = [y[1]]
	for i in 2:n
    		yhat = knn_predict(X[i,:]',Xcond,ycond,k)
    		if y[i] != yhat[1]
			Xcond = vcat(Xcond,X[i,:]')
			push!(ycond,y[i])
    		end
	end
    println("length of cond is: ", length(ycond))
	predict(Xhat) = knn_predict(Xhat,Xcond,ycond,k)
	return GenericModel(predict)
end
