# R script to run the GAMA group finder on G19
library(FoF)
library(celestial)
library(data.table)
library(yaml)

#
# Create the interpolated functions which will be used in the final implementation.
#


redshift = seq(0, 1, by=1e-4)
k_corrections={}
for(i in 1:length(redshift)){
  k_corrections = c(k_corrections, KEcorr(redshift[i])[2])} #K+E corrections

# Creating the redshift to distance modulus and distance modulus to redshift functions.
z_to_mod_matrix = data.frame(redshift = redshift, distance_modulus = cosdistDistMod(redshift, OmegaM=0.25, OmegaL=0.75, H0=100)+k_corrections)
z_to_dmod = approxfun(z_to_mod_matrix[,1], z_to_mod_matrix[,2])
dmod_to_z = approxfun(z_to_mod_matrix[,2], z_to_mod_matrix[,1])

data(LFswml) # reading in the LF as given in the FoF package. (exclusive for GAMA)
cut=-14
tempLFswml = LFswml[LFswml[,2]< cut,c(2,3)]
tempLFswml = cbind(tempLFswml,2*tempLFswml[,2]*(1/sqrt(LFswml[LFswml[,2]< cut,4])))
tempLFswmlLum = tempLFswml
LFswmlfunc = approxfun(tempLFswml[,1],tempLFswml[,2],rule=c(2,2))
LFswmlfuncLum = approxfun(tempLFswml[,1],tempLFswml[,2]*10^(-0.4*tempLFswml[,1]),rule=c(2,2))

#Integrating over the luminosity functions in both magnitude and luminosity
integrated_lf_values = {} #values in the lf function in mag
integrated_lf_values_lum = {} #value in the lf function in luminosities
for(i in 1:length(tempLFswml[,1])){
  integrated_lf_values = c(
    integrated_lf_values, integrate(
      LFswmlfunc, lower=-30, upper=tempLFswml[i,1], subdivisions=1e3)$value)}

for(i in 1:length(tempLFswmlLum[,1])){
  integrated_lf_values_lum = c(
    integrated_lf_values_lum, integrate(
      LFswmlfuncLum, lower=-30, upper=tempLFswml[i,1], subdivisions=1e3, stop.on.error=F)$value)}


minaddswml=min(integrated_lf_values[integrated_lf_values>0])
minaddswmlLum=min(integrated_lf_values_lum[integrated_lf_values_lum>0])
integrated_lf_values[integrated_lf_values==0]=minaddswml
integrated_lf_values_lum[integrated_lf_values_lum==0]=minaddswmlLum
tempLFswml=cbind(tempLFswml,integrated_lf_values)
tempLFswmlLum=cbind(tempLFswmlLum,integrated_lf_values_lum)
LFswmlintfunc=approxfun(tempLFswml[,1],tempLFswml[,4],rule=c(2,2))
LFswmlintfuncLum=approxfun(tempLFswmlLum[,1], tempLFswmlLum[,4], rule=c(2,2))
#
#
#

#
#Randoms stuff:
#
RanCat = fread('gen_ran_out.randoms.csv')
N = 1e4
G09area = skyarea(c(129,141), c(-2,3))
G12area = skyarea(c(174,186), c(-3,2))
G15area = skyarea(c(211.5,223.5), c(-2,3))
gama_fraction_sky = sum(G09area['areafrac'], G12area['areafrac'], G15area['areafrac'])

distfunc_z2D = cosmapfunc('z', 'CoDist', H0=100, OmegaM=0.25, OmegaL=0.75, zrange=c(0,1), step='a', res=N) # redshift to comoving distance
distfunc_D2z = cosmapfunc('CoDist', 'z', H0=100, OmegaM=0.25, OmegaL=0.75, zrange=c(0,1), step='a', res=N) # comoving distance to redshift
RanCat[,'CoDist'] = distfunc_z2D(RanCat[,z])
GalRanCounts = dim(RanCat)[1]/400 # Don't know where this 400 comes from.

#smooth out the histogram of comoving distances
bin = 40
temp = density(RanCat[,CoDist], bw = bin/sqrt(12), from=0, to=2000, n=N, kern='rect')
rm(RanCat)
tempfunc = approxfun(temp$x, temp$y, rule=2) # create a function that maps Distance to frequency
# integrate over the density as a function of distance
tempint = {}
for (colim in seq(0, 2000, len=N)){
  tempint=c(tempint, integrate(tempfunc, colim-bin/2, colim+bin/2)$value)} # not sure I get this exactly

# work out the comoving volume at each bin.
radii = seq(0, 2000, len=N)
volume_of_shells = ((4/3)*pi*(radii + bin/2))**3 - ((4/3)*pi*(radii - bin/2))**3 

RunningVolume = gama_fraction_sky*volume_of_shells
RunningDensity_D = approxfun(temp$x, GalRanCounts*tempint/RunningVolume, rule=2) 
RunningDensity_z = approxfun(distfunc_D2z(temp$x), GalRanCounts*tempint/RunningVolume, rule=2)
#
#
#

############################
# Running the Group Finder #
###########################

# read in the data
g09 = fread('../cut_9.dat')
g09[,'AB_r'] = g09[,Rpetro] - z_to_dmod(g09[,Z])
g09 = as.data.frame(g09)
#I'm just assuming 100% completeness and I should have a look at the way Aaron does the completeness stuff.
#
#
### Reading in the parameters that need to be optimized by the emcee routine ###
params = yaml.load_file('parameters.yml')
#optuse=c(0.05, 23, 0, 0, 0.8, 9.0000, 1.5000, 12.0000)

data(circsamp)
cat=FoFempint(
data=g09, bgal=params$b_gal, rgal=params$r_gal, Eb=params$Eb, Er=params$Er, 
  coscale=T, NNscale=3, groupcalc=T, precalc=F, halocheck=F, apmaglim=19.8, colnames=colnames(g09), 
  denfunc=LFswmlfunc, intfunc=RunningDensity_z, intLumfunc=LFswmlintfuncLum, 
  useorigind=F,dust=0,scalemass=1,scaleflux=1,extra=F,
  MagDenScale=params$mag_den_scale,deltacontrast=params$delta_contrast,deltarad=params$detla_rad,deltar=params$delta_r,
  circsamp=circsamp,Mmax=1e15, zvDmod = z_to_dmod, Dmodvz = dmod_to_z,
  left=129, right=141, top = 3, bottom = -2)


# Write the group_references to file (used for tuning)
# Convert grefs to a data frame to preserve the column structure
grefs_df <- as.data.frame(cat$grefs)
# Write the data frame to file with columns side-by-side
write.table(grefs_df, file = "GAMA_FoFR_run.dat", row.names = FALSE, col.names = FALSE)

