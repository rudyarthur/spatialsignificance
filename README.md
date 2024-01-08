# spatialsignificance
A method for resampling spatial data

Spatial data is usually positively autocorrelated, this is sometimes called [The First Law of Geography or Tobler's Law](https://en.wikipedia.org/wiki/Tobler%27s_first_law_of_geography). This means standard significance tests will give inaccurate results. The presence of autocorrelation means values will cluster and statistics like correlation will tend to be much higher than if there was no autocorrelation. One way to deal with this is to resample while keeping the autocorrelation fixed (which we measure using [Moran's I](https://en.wikipedia.org/wiki/Moran%27s_I)). This code wnables this type of resampling

**This is HIGHLY experimental. Paper coming soon. Please don't use this unless you know what you are doing, which probably means sending me a message to ask**

Install the package with `pip install spatialsignificance`

There are some functions to generate interesting test data in the tests directory

```
import numpy as np
import spatialsignificance as sp
from gen_randomfield import gen_randomfield
from scipy.stats import kendalltau

np.random.seed(123456789)
#generate some test data
size=40
x1 = gen_randomfield(alpha=1.5, size=size)
x2 = gen_randomfield(alpha=1.5, size=size)
```
This generates 2 independent autocorrelated random fields (higher alpha ia more correlated)
![example](https://github.com/rudyarthur/spatialsignificance/blob/main/tests/example.png)

Flatten the image data
```
x1 = x1.flatten()
x2 = x2.flatten()
```
You can then call a function to generate the spatial weight matrix (which cells are neighbours of which)
```
#Figure out the spatial weight matrix
grid = sp.grid_neighbours( size, size, nbrtype="q" )
```
In the case where ou have a list of geometries, e.g. in a geodataframe use
```
grid = sp.poly_neighbours( list_of_shapely_geometries )
```
or 
```
grid = sp.poly_neighbours( list( geo_data_frame['geometry'] ) )
```
Calculate the Moran indices
```
#calculate the moran indices
mi1 = sp.moran(grid, x1)
mi2 = sp.moran(grid, x2)
```
Calculate the statistic of interest (the thing you're significance testing)
```
#compate statistic
kr = kendalltau(x1, x2, alternative="greater")
r = kr.correlation
print("Kendall tau =", r)
print("Scipy p-value :", kr.pvalue)
```
Perform the resampling. We here compare random permutation (ignores autocorrelation) to our method
```
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
```
The output is
```
Kendall tau = 0.05243746091307067
Scipy p-value : 0.0008366722750414914
randomperm : p-value = 0.0008080808080808081
constmoran : p-value = 0.11616161616161616
```
The final value may vary due to seeding the C++ random number generator from the machine's random device. This can be set as a seed argument to the gen_samples function if necessary. The thing to note is that the `constmoran` pvalue is much bigger!
