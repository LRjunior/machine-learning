# Asymmetric aspect model (PLSA) [T. Hofmann 1999]

using SparseArrays
using DelimitedFiles
using Plots
#using PyPlot

function asymmetric_aspect_model(M, Z; maxiter=20)
	D, W = size(M)
	
	@assert all(>=(2), count(>=(1), M, dims=1)) "Document-Word matrix is defective"

	#initialize matrices
	Pzd = rand(Z,D)
	Pzd_new = zeros(Z,D)
	Pwz = rand(W,Z)
	Pwz_new = zeros(W,Z)

	Pdz = rand(D,Z)
	Pdz_new = zeros(D,Z)
	Pzw = rand(Z,W)
	Pzw_new = zeros(Z,W)
	
	#normalize matrices
	s = sum(Pzd, dims=2)
	Pzd = Pzd ./ s
	
	s = sum(Pwz, dims=2)
	Pwz = Pwz ./ s

	s = sum(Pdz, dims=2)
	Pdz = Pdz ./ s

	s = sum(Pzw, dims=2)
	Pzw = Pzw ./ s
	
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
		
		s = sum(Pzd_new, dims=2)
		Pzd = Pzd_new ./ s
		Pzd_new = zeros(Z,D)
		@assert all(isapprox(v, 1.0, atol=1e-4) for v in vec(sum(Pzd, dims=2))) "Normalization of Pzd failed"
		
		s = sum(Pwz_new, dims=2)
		Pwz = Pwz_new ./ s
		Pwz_new = zeros(W,Z)
		@assert all(isapprox(v, 1.0, atol=1e-4) for v in vec(sum(Pwz, dims=2))) "Normalization of Pwz failed"
		
		s = sum(Pdz_new, dims=2)
		Pdz = Pdz_new ./ s
		Pdz_new = zeros(D,Z)
		@assert all(isapprox(v, 1.0, atol=1e-4) for v in vec(sum(Pdz, dims=2))) "Normalization of Pdz failed"
		
		s = sum(Pzw_new, dims=2)
		Pzw = Pzw_new ./ s
		Pzw_new = zeros(Z,W)
		@assert all(isapprox(v, 1.0, atol=1e-4) for v in vec(sum(Pzw, dims=2))) "Normalization of Pzw failed"
		
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

	plot_Pzd = heatmap(Pzd, c=:Greys, xlabel="d", ylabel="z", title="P(z|d)")
	plot_Pwz = heatmap(Pwz, c=:Greys, xlabel="z", ylabel="w", title="P(w|z)")
	plot_Pdz = heatmap(Pdz, c=:Greys, xlabel="z", ylabel="d", title="P(d|z)")
	plot_Pzw = heatmap(Pzw, c=:Greys, xlabel="w", ylabel="z", title="P(z|w)")
	
	plot_matrices = plot(plot_Pzd, plot_Pwz, plot_Pdz, plot_Pzw, layout=(1,4))
	savefig(plot_matrices, "Pzd_Pwz_Pdz_Pzw.pdf")

	open("Pzd.txt", "w") do io
		writedlm(io, Pzd)
	end

	open("Pwz.txt", "w") do io
		writedlm(io, Pwz)
	end

	open("Pdz.txt", "w") do io
		writedlm(io, Pdz)
	end

	open("Pzw.txt", "w") do io
		writedlm(io, Pzw)
	end

	for z in 1:Z
		psp = partialsortperm(Pzw[z,:], 1:top_k, rev=true)
		top_words = collect(zip(Pzw[z,psp], fournews_dictionary[psp,2]))
		println("z=", z)
		for w in top_words
			println(w)
		end
	end
end
