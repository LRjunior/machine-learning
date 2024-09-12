using DelimitedFiles
import PyPlot as plt

begin
	Pzd = readdlm("Pzd.txt", '\t', Float64, '\n')
	Pwz = readdlm("Pwz.txt", '\t', Float64, '\n')
	Pdz = readdlm("Pdz.txt", '\t', Float64, '\n')
	Z = size(Pzd, 1)

	plt.subplot(1,3,1)
	plt.title("P(z|d)")
	plt.xlabel("d")
	plt.ylabel("z")
	plt.yticks(0.5:Z-0.5, 1:Z)
	plt.pcolor(Pzd, cmap=plt.matplotlib.cm.binary, vmin=0, vmax=1)
	plt.colorbar()
	plt.gcf()

	plt.subplot(1,3,2)
	plt.title("P(w|z)")
	plt.xlabel("z")
	plt.ylabel("w")
	plt.xticks(0.5:Z-0.5, 1:Z)
	plt.pcolor(Pwz, cmap=plt.matplotlib.cm.binary, vmin=0, vmax=1)
	plt.colorbar()
	plt.gcf()

	plt.subplot(1,3,3)
	plt.title("P(d|z)")
	plt.xlabel("z")
	plt.ylabel("d")
	plt.xticks(0.5:Z-0.5, 1:Z)
	plt.pcolor(Pdz, cmap=plt.matplotlib.cm.binary, vmin=0, vmax=1)
	plt.colorbar()
	plt.gcf()

	plt.suptitle("Asymmetric aspect model (PLSA)")
	plt.show()
end
