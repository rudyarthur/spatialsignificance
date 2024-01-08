import numpy as np
cimport numpy as cnp
from libc.math cimport exp, ceil, sqrt, pow
from cpython cimport array

from libcpp.random cimport random_device, mt19937, uniform_int_distribution 

cdef random_device dev #doesn't really work anyway as far as I understand...
cdef mt19937 rnd_gen = mt19937(dev())
def reseed(seed):
	cdef long s = seed
	global rnd_gen
	rnd_gen = mt19937(s)
	
cdef uniform_int_distribution[int] rnd_int = uniform_int_distribution[int](0, 1)
		
cdef set_random_int_range(int lo, int hi): #inclusive
	global rnd_int
	rnd_int = uniform_int_distribution[int](lo, hi)

cdef random_int():
	return rnd_int(rnd_gen)	


			
## Not a performance bottleneck so not well optimised, mainly for testing but could be useful for gridded data
## w_ij = nonzero at nbrs[ degs[i]:degs[i+1] ] with value wgt[j]
## The row normalisation makes W asymmetric! e.g.
## zl0 = (w01 w04) . z = (z1/2, z4/3)
## zr0 = z . (w10, w40) = (z1/3, z4/3)
## Allowed neightbour types "r" = rook, "q" = queen
def grid_neighbours(int N, int M, str nbrtype="q", int rownorm=True):

	cdef int T = N*M
	cdef list nbrs = []
	cdef list wgts = []
	cdef list degs = [0]
	
	cdef int i, j, xi, yi, row_sum
	cdef int tot_sum = 0
	cdef double wsum = 0
	for i in range(T):
		xi = i//M
		yi = i%M

		row_sum = 0
		if xi > 0:   
			nbrs.append( (xi-1)*M + yi  ) #n
			wgts.append(1)
			row_sum += 1
			if nbrtype == "q":
				if yi > 0:   
					nbrs.append( (xi-1)*M+yi-1 ) #nw
					wgts.append(1)
					row_sum += 1
				if yi < M-1: 
					nbrs.append( (xi-1)*M+yi+1 ) #ne
					wgts.append(1)
					row_sum += 1
			
		if xi < N-1: 
			nbrs.append( (xi+1)*M + yi  ) #s
			wgts.append(1)
			row_sum += 1
			if nbrtype == "q":
				if yi > 0:   
					nbrs.append( (xi+1)*M+yi-1 ) #sw
					wgts.append(1)
					row_sum += 1
				if yi < M-1: 
					nbrs.append( (xi+1)*M+yi+1 ) #se
					wgts.append(1)
					row_sum += 1
		if yi > 0:   
			nbrs.append( xi*M+yi-1 ) #w
			wgts.append(1)
			row_sum += 1
		if yi < M-1: 
			nbrs.append( xi*M+yi+1 ) #e
			wgts.append(1)
			row_sum += 1

		degs.append( row_sum ) #number of neighbours
		tot_sum += row_sum

		if rownorm:
			for j in range( row_sum ):	wgts[tot_sum - 1 - j] /= row_sum 

	nbrs_ = np.array(nbrs)
	wgts_ = np.array(wgts)
	degs_ = np.cumsum(np.array(degs))


		
	cdef long[:] nbrs_view = nbrs_
	cdef double[:] wgts_view = wgts_
	cdef long[:] degs_view = degs_
		
	return (nbrs_view, wgts_view, degs_view)

def poly_neighbours(polys, rownorm=True): #polys = list of shapely polygons/multipolygons
	
	nbrs = []
	wgts = []
	degs = [0]
	T = len(polys)
	tot_sum = 0
	for i in range(T):
		row_sum = 0
		for j in range(T): #checking everything twice!
			if i!=j and polys[i].intersects( polys[j] ):
				nbrs.append(j)
				wgts.append(1)
				row_sum += 1
				
		degs.append( row_sum ) #number of neighbours
		tot_sum += row_sum

		if rownorm:
			for j in range( row_sum ):	wgts[tot_sum - 1 - j] /= row_sum 

	nbrs_ = np.array(nbrs)
	wgts_ = np.array(wgts)
	degs_ = np.cumsum(np.array(degs))

		
	cdef long[:] nbrs_view = nbrs_
	cdef double[:] wgts_view = wgts_
	cdef long[:] degs_view = degs_

		
	return (nbrs_view, wgts_view, degs_view)
						
		
##x -> z	
def standardise(x):
	return (x - np.mean(x)).flatten()
##z -> x	
def unstandardise(z,x):
	return (z + np.mean(x)).reshape( x.shape )


#zl_a = sum_j w_aj zj
cdef void l_lag(long[:] nbrs, double[:] wgts, long[:] degs, double[:] z, double[:] zl, int T):

	cdef int i, j
	for i in range(T):
		zl[i] = 0
		for j in range( degs[i], degs[i+1] ):
			zl[i] += wgts[j] * z[ nbrs[j] ]

def compute_llag(grid, z):			
	nbrs, wgts, degs = grid	
	cdef int T = z.shape[0]
	cdef double [:] zl = np.zeros( T )

	l_lag(nbrs, wgts, degs, z, zl, T)
	
	return np.array(zl)

#zr_a = sum_j zj w_ja 
#Remember w_ja != w_aj
#The code below only works when all the non-zero elements in a row are equal, e.g. the types of matrices returned by grid_neighbours and poly_neighbours
cdef void r_lag(long[:] nbrs, double[:] wgts, long[:] degs, double[:] z, double[:] zr, int T):

	cdef int i, j
	for i in range(T):
		zr[i] = 0
		for j in range( degs[i], degs[i+1] ): #uses the fact that the neighbour relation is symmetric
			zr[i] += wgts[ degs[nbrs[j]] ] * z[ nbrs[j] ] #weights works because all elements in row are the same so don't have to find the right weight, any will do
			
def compute_rlag(grid, z):			
	nbrs, wgts, degs = grid	
	cdef int T = z.shape[0]
	cdef double [:] zr = np.zeros( T )

	r_lag(nbrs, wgts, degs, z, zr, T)
	
	return np.array(zr)

##super slow, for testing only
#cdef double W( long[:] nbrs,  double[:] wgts,  long[:] degs, int a, int b):
#	cdef int i
#	for i in range( degs[a], degs[a+1] ):
#		if nbrs[i] == b: return wgts[i] 
#	return 0.0

#J = sum_i sum_j w_ij (xi - xbar) (xj - xbar) = sum_i zi zl_i = sum_j zr_j z_j = Z.ZL = ZR.Z
#  = sum_i xi xr_i - xbar ( sum_i (xl_i xr_i) ) + xbar^2 W
def J(grid, z, zr):
	return zr.dot(z)
def Jx(grid, x, xl, xr, xbar, W):
	return xr.dot(x) - xbar*np.sum( xl+xr ) + xbar*xbar*W


#E = sum_i sum_j w_ij (xi - xbar) (xj - xbar) 
# a<->b and w_ab = w_ba = 0
#Eold  za sum waj zj + zb sum wbj zj + (sum_waj zj)za + (sum_wbj zj)zb = za zla + zb zlb + za zra + zb zrb
#Enew  zb sum waj zj + za sum wbj zj + ... = zb zla + za zlb + zb zra + za zrb
#Enew - Eold = (za - zb)( zlb - zla + zrb - zra )
#wab != 0 
#remember zl_a = sum_j w_aj zj
#zla_new = zla - wab zb + wab za = zla +(za-zb)wab#
#zlb_new = zlb - wba za + wba zb = zlb -(za-zb)wba
#zra_new = zra - zb wba + za wba = zra +(za-zb)wba
#zrb_new = zrb - za wab + zb wab = zrb -(za-zb)wab
#gives a term like
#(za-zb)(  zb wab - za wba + zb wba - za wab ) = (za - zb)( wba(zb-za) + wab(zb-za) ) = -(za - zb)(za-zb)(wba + wab)
cdef double deltaJ(long[:] nbrs, double[:] wgts,  long[:] degs, int a, int b, double[:] z, double[:] zl, double[:] zr) noexcept:
	#is a connected to b?
	cdef int i;
	cdef int conn = 0
	for i in range(degs[a],degs[a+1]):
		if nbrs[i] == b: 
			conn = 1
			break
			 
	return (z[a] - z[b]) * ( zr[b] - zr[a] + zl[b] - zl[a] - (z[a] - z[b])*( wgts[ degs[a] ] + wgts[ degs[b] ] )*conn )


#Computation is awful, see paper
cdef double deltaJresample(long[:] nbrs, double[:] wgts,  long[:] degs, int a, double newx, double[:] x, double[:] sum_vars, long T, double Wsum) noexcept:

	cdef double xlxrsum = sum_vars[0]
	cdef double xbar = sum_vars[1]
	cdef double d = (newx - x[a])
	cdef double xbarp = xbar + (d/T)
	cdef double E3 = d*Wsum * (2*xbar + (d/T)) / T
	cdef int i;
	cdef double dxlxrsum = 0
	cdef double E1 = 0
	for i in range( degs[a], degs[a+1] ): 
		dxlxrsum += wgts[ i ] + wgts[ degs[nbrs[i]] ]
		E1 += wgts[ i ]*x[ nbrs[i] ] + wgts[ degs[nbrs[i]] ]*x[ nbrs[i] ]
	E2 = xbarp*(xlxrsum + d*dxlxrsum) - xbar*(xlxrsum)
	E1 *= d 
	
	return E1 - E2 + E3

def call_deltaJ(grid, a, b, z, zl, zr):
	nbrs, wgts, degs = grid
	return deltaJ(nbrs, wgts, degs, a, b, z, zl, zr)
def call_deltaJresample(grid, a, newx, x, sum_vars, W):
	nbrs, wgts, degs = grid
	return deltaJresample( nbrs, wgts, degs, a, newx, x, sum_vars, len(x), W)


#SS = sum_i z_i z_i		
cdef double deltaSSresample(long[:] nbrs, double[:] wgts,  long[:] degs, int a, double newx, double[:] x, double[:] sum_vars, long T) noexcept:

	cdef double xbar = sum_vars[1]
	cdef double d = (newx - x[a])
	#cdef double xbarp = xbar + (d/T)
	cdef double E3 = d * (2*xbar + (d/T))
	#cdef double E2 = d * (xbarp + xbar) #= d* (2*xbar + (d/T))
	cdef double E1 = (newx*newx - x[a]*x[a])
	
	#return E1 - 2*E2 + E3
	return E1 - E3

def call_deltaSSresample(grid, a, newx, x, sum_vars):
	nbrs, wgts, degs = grid
	return deltaSSresample( nbrs, wgts, degs, a, newx, x, sum_vars, len(x))
		
#zl_i = sum_j w_ij zj = w_ia za + w_ib zb + rest
#zl_i' = w_ia zb + w_ib za + rest = zl_i + w_ia zb + w_ib za - w_ia za - w_ib zb
#      = zl_i - w_ia (za-zb) + w_ib (za-zb) 
cdef void updatezlag(long[:] nbrs,  double[:] wgts,  long[:] degs, int a, int b, double[:] z, double[:] zl, double[:] zr) noexcept:

	cdef double d = z[a] - z[b]
	cdef int i
	for i in range( degs[a], degs[a+1] ): 
		zr[ nbrs[i] ] -= d*wgts[ i ]
		zl[ nbrs[i] ] -= d*wgts[ degs[nbrs[i]] ]
		#zl[ nbrs[i] ] -= d*W( nbrs,  wgts,  degs, nbrs[i], a)
	for i in range( degs[b], degs[b+1] ): 
		zr[ nbrs[i] ] += d*wgts[ i ]
		zl[ nbrs[i] ] += d*wgts[ degs[nbrs[i]] ]
		#zl[ nbrs[i] ] += d*W( nbrs,  wgts,  degs, nbrs[i], b)

	cdef double tmp = z[a]
	z[a] = z[b]
	z[b] = tmp

#zl_i = sum_j w_ij zj = w_ia za + rest
#zl_i' = w_ia za' rest = zl_i + w_ia (za' - za) 
#zr_i = sum_j zj w_ji = za wai + rest 
#zr_i' = za' wai + rest = zr_i + w_ai (za' - za) 
cdef void updatexlagresample(long[:] nbrs,  double[:] wgts,  long[:] degs, int a, double newx, double[:] x, double[:] xl, double[:] xr, long T, double[:] sum_vars) noexcept:

	cdef double xlxrsum = sum_vars[0]
	cdef double xbar = sum_vars[1]
		
	cdef double nsum = xlxrsum
	cdef double d = (newx - x[a])
	cdef int i
	for i in range( degs[a], degs[a+1] ): 
		xlxrsum -= xr[ nbrs[i] ] + xl[ nbrs[i] ] 
		xr[ nbrs[i] ] += d*wgts[ i ]
		xl[ nbrs[i] ] += d*wgts[ degs[nbrs[i]] ]
		xlxrsum += xr[ nbrs[i] ] + xl[ nbrs[i] ] 

	x[a] = newx
	xbar += d/T
	
	sum_vars[0] = xlxrsum
	sum_vars[1] = xbar

def call_updatezlag(grid, a, b, z, zl, zr):
	nbrs, wgts, degs = grid
	updatezlag(nbrs, wgts, degs, a, b, z, zl, zr)
def call_updatexlagresample(grid, a, newx, x, xl, xr, sum_vars):
	nbrs, wgts, degs = grid
	updatexlagresample(nbrs, wgts, degs, a, newx, x, xl, xr, len(x), sum_vars)

	
###
# I = (N/W) sum_i sum_j w_ij (xi - xbar) (xj - xbar) / sum_i (xi - xbar)^2 
###
def moran_W(grid):
	return np.sum( grid[1] )
def moran_factor(grid, x):
	return x.shape[0] / ( np.sum( grid[1] ) )
def moran_norm(grid, z):
	return z.shape[0] / ( np.sum( grid[1] ) ) / np.sum(z**2) 
def moran(grid, x):
	z = standardise(x)
	zr = compute_rlag(grid, z)
	return moran_norm(grid, z) * J(grid, z, zr)

	

		
		
###
# I = sum_i sum_j (xi - xbar) (yj - ybar) / sum_j (yj - ybar)^2  \sum_i (xi - xbar)^2
# I = moran_norm E
###				
def pearson_norm(zx, zy):
	return 1 / np.sqrt( (zx*zx).sum() * (zy*zy).sum() )
def pearson_standard(zx, zy):
	return (zx*zy).sum() / np.sqrt( (zx*zx).sum() * (zy*zy).sum() )
def pearson(x, y):
	zx = standardise(x)
	zy = standardise(y)
	return pearson_standard(zx, zy)

	
#Compute the p-value of r given an estimate of the distribution rs = [ estimate1, estimate2, ... ]		
def pval(rs, r, alt="greater", smooth=0):
	larger = (np.array(rs) >= r).sum()
	if alt == "two-tailed" and (len(rs) - larger) < larger: larger = len(rs) - larger
	return (larger + smooth) / (len(rs) + smooth)




#Eold = (t - J)^2 = t*2 - 2tJ + J^2
#Enew = (t - J')^2 = t*2 - 2tJ' + J'^2
#Enew-Eold = 	(- 2tJ' + J'^2) - (-2tJ + J^2)
# = (- 2tJ - 2t dJ + J^2 + 2J dJ + dJ^2) - (-2tJ + J^2)
# = (- 2t dJ  + 2J dJ + dJ^2)
# = dJ(- 2t  + 2J + dJ)
# = -dJ( 2(t - J) - dJ)
cdef long permutise(long[:] nbrs,  double[:] wgts,  long[:] degs, double[:] z, double[:] zl, double[:] zr, double J, double targ, long maxits, int T, double norm, double eps, double eta) noexcept:
	
	cdef double dJ, dE, dI, Iold
	cdef int a, b
	cdef long its = 0
	cdef double I = norm*J
	Iold = I
	
	cdef int acceptall = 1
		
	while (abs(targ - I) > eps or acceptall) and (its < maxits):

		#swap
		a = random_int()
		b = random_int()
		if a == b: continue
		
		#new energy
		dJ = deltaJ(nbrs, wgts, degs, a, b, z, zl, zr)
		dI = norm*dJ
		dE = dI*( 2*(targ - I) - dI )

		if (acceptall and dI > 0) or (not acceptall and dE >= 0): 
			J += dJ
			I += dI
			updatezlag(nbrs,  wgts,  degs, a, b, z, zl, zr)
			

		if acceptall and (its > 0 and its%T == 0): 
			if I > 2*targ or I - Iold < eta: acceptall = 0
			Iold = I
		
		its += 1 
	#print("finished in", its, "with diff", abs(targ - I), "I=", I, "targ=", targ, "Imax = ", Iold )
	return its

cdef long sampalise(long[:] nbrs,  double[:] wgts,  long[:] degs, 
double[:] sample, double[:] x, double[:] xl, double[:] xr, 
double J, double SS, 
double[:] sum_vars, double factor, long T, double Wsum, 
double targ, long maxits, double eps, double eta) noexcept:
	
	cdef double dJ, dSS, dI, dE, newx, oldx, Iold 
	cdef int a, b
	cdef long its = 0
	cdef double I = factor*J/SS
	Iold = I

	cdef int acceptall = 1
	
	
	while (abs(targ - I) > eps or acceptall) and (its < maxits):

		#resample
		a = random_int() #location
		b = random_int() #value
		if a == b: continue
		newx = sample[b]
		oldx = x[a]
		
		dJ = deltaJresample(nbrs, wgts, degs, a, newx, x, sum_vars, T, Wsum)
		dSS = deltaSSresample(nbrs, wgts, degs, a, newx, x, sum_vars, T)
		dI = factor*(  (J + dJ)/(SS + dSS)  -  (J/SS) )
		dE = dI*( 2*(targ - I) - dI )


		if (acceptall and dI >= 0) or (not acceptall and dE >= 0): 
			J += dJ
			SS += dSS
			I += dI	
			updatexlagresample(nbrs, wgts, degs, a, newx, x, xl, xr, T, sum_vars)

		
		if acceptall and (its > 0 and its%T == 0): 
			if I > 2*targ or I - Iold < eta: acceptall = 0
			Iold = I
			
		its += 1 
	#print("finished in", its, "with diff", abs(targ - I), "I=", I, "targ=", targ, "Imax = ", Iold )
	return its
	
		
def gen_samples(grid, x, Nperms, target=1, maxits=10000000, seed=None, eps=1e-7, eta=1e-7, strategy="sample"):
	
	nbrs_, wgts_, degs_ = grid
	cdef long[:] nbrs = nbrs_
	cdef double[:] wgts = wgts_
	cdef long[:] degs = degs_

	cdef int T = len(x)
	if seed: reseed(seed)
	set_random_int_range(0,T-1)
	cdef long mits = maxits
	cdef int repeatc = 0
	permutations = []
	
	cdef double [:] xl = np.zeros( T )
	cdef double [:] xr = np.zeros( T )
	cdef double [:] xp = np.zeros( T )

	cdef double [:] z = standardise(x)
	cdef double [:] zl = np.zeros( T )
	cdef double [:] zr = np.zeros( T )
	cdef double [:] zp = np.zeros( T )
	
	cdef double S, J, s, temp, its
	cdef double factor = moran_factor(grid, x )
	cdef double Wsum = moran_W(grid)
	
	cdef double norm = moran_norm(grid, np.array(z) )
	cdef int tryperm = 0
	
	while len(permutations) < Nperms and tryperm < 10*Nperms:

		##start from a permutation
		xp = np.random.permutation( x ) 

		if strategy == "permute":
			zp = standardise(xp)
			zl = np.zeros(T)
			zr = np.zeros(T)
			l_lag(nbrs, wgts, degs, zp, zl, T)
			r_lag(nbrs, wgts, degs, zp, zr, T)
			
			J = np.array(zr).dot( np.array(zp) )
			
			its = permutise(nbrs,  wgts,  degs, zp, zl, zr, J, target, maxits, T, norm, eps, eta)
			xp = unstandardise(zp,x)
		else:
			xl = np.zeros(T)
			xr = np.zeros(T)
			l_lag(nbrs, wgts, degs, xp, xl, T)
			r_lag(nbrs, wgts, degs, xp, xr, T)
		
			xlxrsum = np.sum( np.sum(xl) + np.sum(xr) )
			xbar = np.mean(xp)
			sum_vars = np.array([xlxrsum, xbar])
			J = Jx(grid, np.array(xp), np.array(xl), np.array(xr), xbar, Wsum)
			SS = np.sum( (xp-xbar)**2 )
			
			its = sampalise(nbrs,  wgts,  degs, x, xp, xl, xr, J, SS, sum_vars, factor, T, Wsum, target, maxits, eps, eta)
			
		if its < maxits: 
			permutations.append( np.array(xp) )			
			#print("found", moran(grid, xp), "with target", targ, "after", its)
			
		tryperm += 1
		
	if len(permutations) != Nperms:
		print("only found", len(permutations), "try increasing maxits")	
	return permutations	
	
	

