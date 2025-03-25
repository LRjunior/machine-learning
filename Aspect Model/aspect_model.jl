# Asymmetric aspect model (PLSA) [T. Hofmann 1999]

using SparseArrays
using DelimitedFiles
using Plots
#using PyPlot

function normalize!(A::Matrix, B::Matrix)
	#normalize matrices
	s = sum(A, dims=2)
	B = A ./ s
	A = zeros(size(A))
	@assert all(isapprox(v, 1.0, atol=1e-4) for v in vec(sum(B, dims=2))) "Normalization failed"
	return A, B
end

function write_results(matrix::Matrix, filename::String)
	open(filename, "w") do io
		writedlm(io, matrix)
	end
end

function asymmetric_aspect_model(M, Z::Int; maxiter::Int=20)
	D, W = size(M)
	
	@assert all(>=(2), count(>=(1), M, dims=1)) "Document-Word matrix is defective"

	#initialize matrices
	Pzd = rand(Z,D)
	Pwz = rand(W,Z)
	Pdz = rand(D,Z)
	Pzw = rand(Z,W)
	Pzd_new = zeros(Z,D)
	Pwz_new = zeros(W,Z)
	Pdz_new = zeros(D,Z)
	Pzw_new = zeros(Z,W)
	
	#normalize matrices
	Pzd,Pzd_new = normalize!(Pzd,Pzd_new)
	Pwz,Pwz_new = normalize!(Pwz,Pwz_new)
	Pdz,Pdz_new = normalize!(Pdz,Pdz_new)
	Pzw,Pzw_new = normalize!(Pzw,Pzw_new)
	
	#perform iterative EM algorithm
	log_likelihood_list = []
	iter = 0
	while iter < maxiter
		for (d,w,n) in zip(findnz(M)...)
			denom = sum(Pzd[z,d] * Pwz[w,z] for z in 1:Z)
			@assert denom >= 0 "denominator is less than zero"
			
			for z in 1:Z
				Pzdw = if denom == 0 0 else (Pzd[z,d] * Pwz[w,z]) / denom end
				part = n * Pzdw
				Pzd_new[z,d] += part
				Pwz_new[w,z] += part
				Pdz_new[d,z] += part
				Pzw_new[z,w] += part
			end
		end
		
		Pzd_new,Pzd = normalize!(Pzd_new,Pzd)
		Pwz_new,Pwz = normalize!(Pwz_new,Pwz)
		Pdz_new,Pdz = normalize!(Pdz_new,Pdz)
		Pzw_new,Pzw = normalize!(Pzw_new,Pzw)
		
		loglikelihood = 0
		for (d,w,n) in zip(findnz(M)...)
			s = sum(Pzd[z,d] * Pwz[w,z] for z in 1:Z)
			@assert s > 0 "sum should be greater than zero"
			loglikelihood += n * log2(s)
		end
		
		@assert isempty(log_likelihood_list) ? true : loglikelihood >= log_likelihood_list[end] "loglikehood is not rising"
		
		push!(log_likelihood_list, loglikelihood)
		iter += 1
	end
	return log_likelihood_list, Pzd, Pwz, Pdz, Pzw
end


begin
	fournews_dictionary = readdlm("4news_dictionary.txt")
	fournews_matrix = sparse(transpose(readdlm("4news.txt", ' ', Int, '\n')))
	top_k = 10
	Z = 4
	loglikelihood_list, Pzd, Pwz, Pdz, Pzw = @time asymmetric_aspect_model(fournews_matrix, Z, maxiter=100)

	plot_loglikelihood = plot(loglikelihood_list, xlabel="iteration", ylabel="log-likelihood", title="Log-likelihood")
	savefig(plot_loglikelihood, "log_likelihood.pdf")

	plot_Pzd = heatmap(Pzd, c=:Greys, xlabel="d", ylabel="z", title="P(z|d)", clims=(0.0,1.0))
	plot_Pwz = heatmap(Pwz, c=:Greys, xlabel="z", ylabel="w", title="P(w|z)", clims=(0.0,1.0))
	plot_Pdz = heatmap(Pdz, c=:Greys, xlabel="z", ylabel="d", title="P(d|z)", clims=(0.0,1.0))
	plot_Pzw = heatmap(Pzw, c=:Greys, xlabel="w", ylabel="z", title="P(z|w)", clims=(0.0,1.0))
	
	plot_matrices = plot(plot_Pzd, plot_Pwz, plot_Pdz, plot_Pzw, layout=(1,4))
	savefig(plot_matrices, "Pzd_Pwz_Pdz_Pzw.pdf")

	write_results(Pzd, "Pzd.txt")
	write_results(Pwz, "Pwz.txt")
	write_results(Pdz, "Pdz.txt")
	write_results(Pzw, "Pzw.txt")
	
	for z in 1:Z
		psp = partialsortperm(Pzw[z,:], 1:top_k, rev=true)
		top_words = collect(zip(Pzw[z,psp], fournews_dictionary[psp,2]))
		println("z=", z)
		for w in top_words
			println(w)
		end
	end
end
