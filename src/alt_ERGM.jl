# storing this code here until I have a better use for it
function update_ergm(current_params, graph, graphstats, K, err)
	proposed_ergm = copy(current_params) #rand(MvNormal(current_params, sigma))
	for i in 1:length(current_params)
		proposed_ergm[i] = rand(Normal(proposed_ergm[i], err[i]))
	end
	auxstats = rgraph(graph, proposed_ergm, graphstats, K)
	a = log(rand())
	MHratio = ll(proposed_ergm, current_params, graphstats, auxstats)
	if a < MHratio
		return proposed_ergm, 1, MHratio
	else
		return current_params, 0, MHratio
	end
end

function ergm(g::SimpleDiGraph{Int}, startvec, err, num_iter, num_thin, adapt_iter, K)
	ergm_params = Array{Float64}(undef, length(startvec), num_iter)
	ergm_params[:, 1] = startvec

	current_params = deepcopy(startvec)
	ergm_count, ergm_ratio = 0, 0.0

	running_ergm_stats = CovMatrix()
	running_acceptance_ergm = Mean()
	fit!(running_ergm_stats, startvec)
	fit!(running_acceptance_ergm, 1)

	graphstats = s(g)
	current_graphstats = copy(graphstats)

	for mcmciter = 2:num_iter
		for thiniter = 1:num_thin
			current_params, ergm_count, ergm_ratio = update_ergm(
				current_params, g, current_graphstats, K, err
			)
			fit!(running_ergm_stats, current_params)
			fit!(running_acceptance_ergm, ergm_count)
		end
		println("Iter: ", mcmciter, " ERGM: ", round.(current_params, digits=3))
		ergm_params[:, mcmciter] = current_params
	end
	return ergm_params
end

"""
pl_init = coef(psuedo_likelihood(g, 2))
prop_vec = diagm(vcat(0.01, fill(0.001, length(pl_init) - 1)))
AMH_scaling_factor = 2.38^2 / length(pl_init)
ergm(g, pl_init, prop_vec, AMH_scaling_factor, 5000, 75, 100, 5)


pl = psuedo_likelihood(g, 2)
pl_init = coef(pl)
err = stderror(pl)
ergm(g, pl_init, err, 5000, 75, 100, 5)
"""
