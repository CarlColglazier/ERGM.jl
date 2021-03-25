# Exponential-family random graph model starter code.
using LightGraphs
using CSV
using CategoricalArrays
using Distributions
using Random
using LinearAlgebra
using StaticArrays
using Distributions, StatsFuns, OnlineStats

#--- utils ---
function loadcsv_graph(f)
	file = f
	df = CSV.read(uri_peers_file, limit=100_000)
	g = DiGraph(size(unique(df.Source))[1])
	codes = levelcode.(categorical(df))
	codemap = Dict(unique(codes.Source) .=> unique(df.Source))
	for row in eachrow(codes)
		add_edge!(g, row.Source, row.Target)
	end
	return g
end

"""
For each vertex in the graph, calculate the probability
that an outgoing edge is reciprocated.
"""
function mutual(g::SimpleDiGraph{Int})
	m = adjacency_matrix(g)
	return sum(m .* m')
end

"""
For a set of three nodes i, j, k, how many paths
can we draw such that i->j->k?
"""
function cyclic_triads(g::SimpleDiGraph{Int})
	sum = 0
	m = adjacency_matrix(g)
	for i in 1:nv(g)
		for j in 1:nv(g)
			for k in 1:nv(g)
				sum += m[i, j] * m[j, k] * m[k, i]
			end
		end
	end
	return sum
end

#--- estimator functions ---
function ss(g::SimpleDiGraph{Int})
	return SA_F64[
		ne(g),
		mutual(g),
		#cyclic_triads(g),
		#reciprocity(g)
	]
end

function permute!(g::SimpleDiGraph{Int}, x_r::Int, y_r::Int)
	if has_edge(g, x_r, y_r)
		rem_edge!(g, x_r, y_r)
	else
		add_edge!(g, x_r, y_r)
	end
	return nothing
end

function weight(g::SimpleDiGraph{Int}, θ::Array{Float64})
	return exp.(sum(θ .* ss(g)))
end

function δ1(g::SimpleDiGraph{Int}, x::Int, y::Int)
	if !has_edge(g, x, y)
		add_edge!(g, x, y)
		dplus = ss(g)
		rem_edge!(g, x, y)
		return dplus - ss(g)
	else
		rem_edge!(g, x, y)
		dminus = ss(g)
		add_edge!(g, x, y)
		return ss(g) - dminus
	end
end

function weight_diff(g1::SimpleDiGraph{Int}, g2::SimpleDiGraph{Int}, θ::Array{Float64})
	s1 = s(g1)
	s2 = s(g2)
	return exp.(sum(θ .* (s1 - s2)))
end


function rgraph_mcmc(
		g::SimpleDiGraph{Int}, p::Array{Float64};
		n_samples::Int=1024, n_i::Int=512, burnin::Int=512
	)
	weights = zeros(Float64, n_samples) .+ 1
	s_arr = Array{Float64}(undef, n_samples, size(p, 1))
	samp = SimpleDiGraph(nv(g), ne(g)) # a random graph with the same density
	# This could also come from the observed network.
	this_w = weight(samp, p)
	iters = 0
	i = 1
	while i <= (n_samples + burnin)
		iters += 1
		x_r = rand(1:nv(g))
		y_r = rand(1:nv(g))
		permute!(samp, x_r, y_r)
		this_s = ss(samp)
		new_w = exp.(sum(p .* this_s)) #weight(s, p)
		if rand() < (new_w/this_w)
			this_w = new_w
		else
			# undo
			permute!(samp, x_r, y_r)
		end
		if iters % n_i == 0
			if i > burnin
				weights[i-burnin] = new_w
				for j in 1:size(p, 1) # can prob just assign the array.
					s_arr[i-burnin, j] = this_s[j]
				end
			end
			i += 1
		end
	end
	return weights, s_arr
end

function p_rgraph_mcmc(
		g::SimpleDiGraph{Int}, p::Array{Float64};
		n_samples::Int=128, n_i::Int=128, burnin::Int=256,
		n_threads::Int=Threads.nthreads()
	)
	w_arr = Array{Float64}(undef, n_samples*n_threads, 1)
	t_arr = Array{Float64}(undef, n_samples*n_threads, size(p, 1))
	Threads.@threads for i in 0:(n_threads-1)
		r, t = rgraph_mcmc(g, p, n_samples=n_samples, n_i=n_i, burnin=burnin)
		t_arr[1+i*n_samples:(i+1)*n_samples, :] = t
		w_arr[1+i*n_samples:(i+1)*n_samples] = r
		end
	return w_arr, t_arr
end

# start with -4.17, 3.35
# end wiht -3.5, 3.26
function p_fit(
		g::SimpleDiGraph{Int}, coeffs::Array{Float64}, n_iter::Int;
		alpha::Float64=0.1
	)
	ps = 0.0
	all_coeffs = [coeffs]
	for i in 1:n_iter
		coef = copy(last(all_coeffs))
		for j in 1:length(coef)
			coef[j] = rand(Normal(coef[j], alpha))
		end
		r, t = p_rgraph_mcmc(g, coef, n_samples=1024, n_i=64)
		p = weight(g, coef) / sum(r)
		if rand() < (p / ps)
			push!(all_coeffs, coef)
			ps = p
		else
			push!(all_coeffs, last(all_coeffs))
		end
		println("p for ", coef, " ", p, " at ", i, " of ", n_iter)
	end
	return all_coeffs
end

### Copying from another github repo.
function rgraph(g::SimpleDiGraph{Int}, p::Array{Float64}, graphstats, K)
	q = deepcopy(g)
	n = nv(g)
	x = copy(graphstats)
	for k = 1:K
		for i = 1:n
			for j = 1:n
				i == j && continue
				deltastats = δ(q, i, j)
				if log(rand()) < dot(p, deltastats)
					permute!(q, i, j)
					x += deltastats
				end
			end
		end
	end
	return x
end

function fit(g::SimpleDiGraph{Int}, coeffs::Array{Float64}, err, n_iter::Int)
	alpha = 0.1
	ps = 0.0
	all_coeffs = [coeffs]
	d = Normal(0, alpha)
	for i in 1:n_iter
		coef = copy(last(all_coeffs))#
		for j in 1:length(coef)
			coef[j] = rand(Normal(coef[j], err[j]))
		end
		p = weight(g, coef) / mysample(g, coef)
		if rand() < (p / ps)
			push!(all_coeffs, coef)
			ps = p
		else
			push!(all_coeffs, last(all_coeffs))
		end
		println("p for ", coef, " ", p, " at ", i, " of ", n_iter)
	end
	return all_coeffs
end

#g = SimpleDiGraph(Matrix(CSV.read("../data/padgett.csv", delim=" ", header=false)))
#fit(g, [0.0, 0.0], 100)

## See https://github.com/adamhaber/ergmjl
## for the basis for a lot of these functions.
## Eventually, I predict Julia could build some good
## tools for ERGMs, but it's not there quite yet.
using GLM
function psuedo_likelihood(g::SimpleDiGraph{Int}, n_params=2)
	n = nv(g)
	X = Array{Float64}(undef, n * (n - 1), n_params)
	y = Array{Bool,1}(undef, n * (n - 1))
	counter = 1
	for i in 1:n
		for j in 1:n
			i == j && continue
			y[counter] = has_edge(g, i, j)
			X[counter,:] = δ1(g, i, j)
			counter += 1
		end
	end
	return GLM.fit(GeneralizedLinearModel, X, y, Binomial(), LogitLink())
end

function ll(proposed_graph_params, current_params, graphstats, auxstats)
	ll = 0.0
	ll += dot(graphstats, proposed_graph_params) - dot(graphstats, current_params)
	ll += dot(auxstats, current_params) - dot(auxstats, proposed_graph_params)
	ll += loglikelihood(Normal(0, 1), proposed_graph_params) - loglikelihood(Normal(0, 1), current_params)
	return ll
end

# take community cluster
# snapshot ? just focus on a limited time period.
#   long-run? use quest.
#   for ERGMs, look at different algorithms.
#   we included a couple papers for doing this.
#   one idea is snowball sampling (keynet).
#   another idea is a clustering computing approach
#   throw the calclution to different computational clusters and
#   combine them.
