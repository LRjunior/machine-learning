# A Simple Probabilistic Algorithm for Detecting Community Structure in Social Networks [Ren et al. 2007]

using LaTeXStrings
using Plots; pyplot()

zachary_network = [(2,1),(3,1),(3,2),(4,1),(4,2),(4,3),(5,1),(6,1),(7,1),(7,5),(7,6),(8,1),(8,2),(8,3),(8,4),(9,1),(9,3),(10,3),(11,1),(11,5),(11,6),(12,1),(13,1),(13,4),(14,1),(14,2),(14,3),(14,4),(17,6),(17,7),(18,1),(18,2),(20,1),(20,2),(22,1),(22,2),(26,24),(26,25),(28,3),(28,24),(28,25),(29,3),(30,24),(30,27),(31,2),(31,9),(32,1),(32,25),(32,26),(32,29),(33,3),(33,9),(33,15),(33,16),(33,19),(33,21),(33,23),(33,24),(33,30),(33,31),(33,32),(34,9),(34,10),(34,14),(34,15),(34,16),(34,19),(34,20),(34,21),(34,23),(34,24),(34,27),(34,28),(34,29),(34,30),(34,31),(34,32),(34,33)]

function ren_community_model(M, C, maxiter=20)
	N = 0
	for (a,b) in M
		N = max(N, a, b)
	end
	@assert N > 0 "N must be greater than zero"
	
	#ensure matrix is symmetric
	for (a,b) in M
		if !((b,a) in M)
			push!(M, (b,a))
		end
	end

	#check matrix there must not be elements on diagonal
	element_on_diagonal = false
	for (a,b) in M
		if a == b
			element_on_diagonal = true
		end
	end
	@assert !element_on_diagonal "element on diagonal" #there should not be any non-zero element on diagonal

	#check if matrix is not defective, each row must contain at least one non-zero element 
	m = Set()
	for (a,b) in M
		union!(m, a)
	end
	#isempty(intersect([1,2,3],[4,5,6]))
	
	i = 1
	defective = false
	for a in m
		if !(i in m)
			defective = true
			break
		end
		i += 1
	end
	@assert !defective "matrix is defective"
	#println("defective:", defective)
	
	#initialize matrices
	P_pi = rand(C)
	P_beta = rand(C, N)
	P_pi_new = zeros(C)
	P_beta_new = zeros(C, N)

	#normalize matrices
	
	s = 0
	for c in 1:C
		s += P_pi[c]
	end
	for c in 1:C
		P_pi[c] /= s
	end
	
	s = 0
	for c in 1:C
		s += P_pi[c]
	end
	@assert isapprox(s, 1.0)
	
	for c in 1:C
		s = 0
		for n in 1:N
			s += P_beta[c,n]
		end
		for n in 1:N
			P_beta[c,n] /= s
		end
	end	
	
	for c in 1:C
		s = 0
		for n in 1:N
			s += P_beta[c,n]
		end
		@assert isapprox(s, 1.0)
	end
	
	#perform iterative EM algorithm
	loglikelihood_list = []
	iter = 0
	while iter < maxiter
		for (i,j) in M
			denom = 0
			for c in 1:C
				denom += P_pi[c] * P_beta[c,i] * P_beta[c,j]
			end
			@assert denom >= 0 "denominator is less than zero"
			
			for c in 1:C
				if denom == 0
					P_ei = 0
				else
					P_ei = (P_pi[c] * P_beta[c,i] * P_beta[c,j]) / denom
				end
				P_pi_new[c] += P_ei
				P_beta_new[c,i] += P_ei
			end
		end
		
		s = 0
		for c in 1:C
			s += P_pi_new[c]
		end
		@assert s > 0 "sum is zero"
			
		for c in 1:C
			P_pi[c] = P_pi_new[c] / s
			P_pi_new[c] = 0
		end
		@assert isapprox(sum(P_pi), 1.0) "normalization error"
		
		for c in 1:C
			s = 0
			for n in 1:N
				s += P_beta_new[c,n]
			end
			@assert s > 0 "sum must be greater than zero"
			
			for n in 1:N
				P_beta[c,n] = P_beta_new[c,n] / s
				P_beta_new[c,n] = 0
			end
		end
		
		loglikelihood = 0
		for (i,j) in M
			s = 0
			for c in 1:C
				s += P_pi[c] * P_beta[c,i] * P_beta[c,j]
			end
			@assert s > 0 "sum must be greater than zero"
			loglikelihood += log2(s)
		end
		
		@assert isempty(loglikelihood_list) ? true : loglikelihood >= loglikelihood_list[end] "loglikehood is not rising"
		
		push!(loglikelihood_list, loglikelihood)
		iter += 1
	end
	return (loglikelihood_list, P_pi, P_beta)
end

begin
	C = 2
	loglikelihood_list, P_pi, P_beta = @time ren_community_model(zachary_network, C)

	plot!(loglikelihood_list)
	p1 = heatmap(Matrix(P_pi')', c=:Greys, ylabel="Community", title=L"π_c", clims=(0.0,1.0))
	p2 = heatmap(P_beta, c=:Greys, xlabel="Node", ylabel="Community", title=L"β_{c,i}", clims=(0.0,1.0))
	plot(p1, p2)
end
