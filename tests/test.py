import sys
sys.path.append('../src/')
import numpy as np
import spatialsignificance as sp
from gen_randomfield import gen_randomfield
from scipy.stats import kendalltau

np.random.seed(123456789)
#generate some test data
size=40
x1 = gen_randomfield(alpha=1.5, size=size)
x2 = gen_randomfield(alpha=1.5, size=size)

##plot it if you want
import matplotlib.pyplot as plt
plt.imshow(x1, cmap="RdBu")
plt.savefig("example.png")
plt.show()
plt.close()

#Methods expect 1d lists
x1 = x1.flatten()
x2 = x2.flatten()

#Figure out the spatial weight matrix
grid = sp.grid_neighbours( size, size, nbrtype="q" )
#If you have a list of shapely polygons use
#grid = sp.poly_neighbours( list_of_geometries )

#calculate the moran indices
mi1 = sp.moran(grid, x1)
mi2 = sp.moran(grid, x2)

#compate statistic
kr = kendalltau(x1, x2, alternative="greater")
r = kr.correlation
print("Kendall tau =", r)
print("Scipy p-value :", kr.pvalue)
#compare random permutations to random permutations at fixes I
Nperms = 100
for method in ["randomperm", "constmoran"]:
	
	if method == "randomperm":
		perms1 = [ np.random.permutation(x1) for i in range(Nperms) ]
		perms2 = [ np.random.permutation(x2) for i in range(Nperms) ]	
	elif method == "constmoran":
		perms1 = sp.gen_samples(grid, x1, Nperms, target=mi1)
		perms2 = sp.gen_samples(grid, x2, Nperms, target=mi2)

	##compare all perms1 against amm perms2
	rs = []
	for ii in range(len(perms1)):
		px1 = perms1[ii]
		for j in range(ii+1,len(perms2)):
			px2 = perms2[j]
			rs.append(  kendalltau(px1,px2).correlation )
	rs = np.array(rs)
	
	print(method, ": p-value =", sp.pval(rs, r, alt="greater") )



