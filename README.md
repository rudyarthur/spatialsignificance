# spatialsignificance
A method for resampling spatial data

Spatial data is usually positively autocorrelated, this is sometimes called [The First Law of Geography or Tobler's Law](https://en.wikipedia.org/wiki/Tobler%27s_first_law_of_geography). This means standard significance tests will give inaccurate results. The presence of autocorrelation means values will cluster and statistics like correlation will tend to be much higher than if there was no autocorrelation. One way to deal with this is to resample while keeping the autocorrelation fixed (which we measure using [Moran's I](https://en.wikipedia.org/wiki/Moran%27s_I)). The whole method is described in [this paper](https://onlinelibrary.wiley.com/doi/full/10.1111/gean.12417).

## Installation
**[pip coming soon...]**

You need Cython>=3 and numpy >=2

`python3 -m pip install Cython`

`python3 -m pip install numpy`

you sould also install the usual python data analysis stuff to run the tests

`python3 -m pip install scipy`

`python3 -m pip install matplotlib`

then checkout the git repo

`git clone https://github.com/rudyarthur/spatialsignificance.git`

and compile it

`python3 setup.py build_ext --inplace`

this will build the library in the `src` directory of the checked out repository. To use the library you have to tell python where to look for it e.g. at the top of your program write
```
import sys
sys.path.append('/your/path/to/the/.so/file')
import spatialsignificance as sp
```

## Using
There are some functions to generate interesting test data in the tests directory

```
import numpy as np
import sys
sys.path.append('../src')
import spatialsignificance as sp
from gen_randomfield import gen_randomfield
from scipy.stats import kendalltau

np.random.seed(123456789)
#generate some test data
size=40
x1 = gen_randomfield(alpha=1.5, size=size)
x2 = gen_randomfield(alpha=1.5, size=size)
```
This generates 2 independent autocorrelated random fields (higher alpha is more correlated) which look something like below
![example](https://github.com/rudyarthur/spatialsignificance/blob/main/tests/example.png)

To perform a significance test, first, flatten the image data
```
x1 = x1.flatten()
x2 = x2.flatten()
```
Then call a function to generate the spatial weight matrix (which cells are neighbours of which).
```
#Figure out the spatial weight matrix
# Allowed neightbour types "r" = rook, "q" = queen
grid = sp.grid_neighbours( size, size, nbrtype="q" )
```
In the case where you have a list of geometries, e.g. in a geodataframe use
```
grid = sp.poly_neighbours( list_of_shapely_geometries )
```
or 
```
grid = sp.poly_neighbours( list( geo_data_frame['geometry'] ) )
```
Calculate the Moran indices of your data
```
#calculate the moran indices
mi1 = sp.moran(grid, x1)
mi2 = sp.moran(grid, x2)
```
Calculate the statistic of interest (the thing you're significance testing). Here I'm computing the [Kendall's Tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) but you can swap this with any function of the two data sets e.g. pearsonr, spearmanr, x1/x2, ... 
```
#compare statistic
kr = kendalltau(x1, x2, alternative="greater")
r = kr.correlation
print("Kendall tau =", r)
print("Scipy p-value :", kr.pvalue)
```
Perform the resampling. We here compare random permutation (ignores autocorrelation) to our method. If you're not interested in this comparison, leave out the `randomperm` part
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
The final value may vary due to seeding the C++ random number generator from the machine's random device. This can be set as a seed argument to the gen_samples function if necessary. The thing to note is that the `constmoran` pvalue is much bigger than the random permutation sampling!
