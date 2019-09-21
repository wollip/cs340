include("misc.jl") # Includes GenericModel typedef

function naiveBayes(X,y)
	# Implementation of naive Bayes classifier for binary features

	(n,d) = size(X)

  # Compute number of classes, assuming y in {1,2,...,k}
  k = maximum(y)

  # We will store p(y(i) = c) in p_y(c)
  counts = zeros(k)
  for i in 1:n
    counts[y[i]] += 1
  end
  p_y = counts ./ n

  # We will store p(x(i,j) = 1 | y(i) = c) in p_xy(1,j,c)
  # We will store p(x(i,j) = 0 | y(i) = c) in p_xy(2,j,c)
    trues = zeros(d, k)
    falses = zeros(d, k)
    for i in 1:n
        trues[:, y[i]] .+= X[i, :]
        falses[:, y[i]] .+= (1 .- X[i, :])
    end
    
    p_xy = zeros(2,d,k)

    top = trues ./ reshape(counts, 1, k)
    bot = sum(trues, dims = 2) ./ n
    p_xy[1, :, :] = top ./ bot
#     p_xy[1, :, :] = top

    top = falses ./ reshape(counts, 1, k)
    bot = sum(falses, dims = 2) ./ n
    p_xy[2, :, :] = top ./ bot
#     p_xy[2, :, :] = top

    replace!(p_xy, NaN=>0)

  function predict(Xhat)
    (t,d) = size(Xhat)
    yhat = zeros(t)

    for i in 1:t
      # p_yx = p_y*prod(p_xy) for the appropriate x and y values
      p_yx = copy(p_y)
      for j in 1:d
        if Xhat[i,j] == 1
          for c in 1:k
            p_yx[c] *= p_xy[1,j,c]
          end
        else
          for c in 1:k
            p_yx[c] *= p_xy[2,j,c]
          end
        end
      (~,yhat[i]) = findmax(p_yx)
      end
    end
    return yhat
  end

	return GenericModel(predict)
end
