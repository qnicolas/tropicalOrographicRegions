import numpy as np
import xarray as xr
try:
    from xarray.ufuncs import cos, deg2rad
except ModuleNotFoundError:
    from numpy import cos
    def deg2rad(deg):
        return deg*np.pi/180
    
from scipy import special

###################################################################################################
###################################### BOXES, SELECTION ###########################################
###################################################################################################

def sel_box(var,box,lon='longitude',lat='latitude',lat_increase=False):
    if lat_increase:
        return var.sel({lon:slice(box[0],box[1]),lat:slice(box[2],box[3])})
    else:
        return var.sel({lon:slice(box[0],box[1]),lat:slice(box[3],box[2])})

def sel_months(ds,months):
    try: 
        return ds.sel(month = np.in1d( ds['month'], months))
    except KeyError:
        return ds.sel(time = np.in1d( ds['time.month'], months))

def sel_box_months(var,box,months,lon='longitude',lat='latitude',lat_increase=False):
    window = sel_box(var,box,lon,lat,lat_increase)
    if "month" in var.dims:
        window=sel_months(window,months).mean('month')
    return window


###########################################################################################################
########################################### SPATIAL FUNCTIONS #############################################
###########################################################################################################

def spatial_mean(ds,box=None,mask=None,lat='latitude',lon='longitude'):
    """Given a dataset with at least two spatial dimensions, compute a spatial mean within a specified region
    defined by a mask, inside a given box
        - variable = ND xarray.dataarray. Spatial dimensions names are given by the arguments 'lat' and 'lon'
        - mask = None, or 2D xarray.dataarray of 0s and 1s. Must have same grid and dimension names as 'variable'.
        - box = None, or list of four items, [lon1, lon2, lat1, lat2]
        - lat = str, name of the latitude coordinate
        - lon = str, name of the longitude coordinate
    """
    if type(ds)==int:
        return ds
    coslat = cos(deg2rad(ds[lat]))
    weight = coslat.expand_dims({lon:ds[lon]}) / coslat.mean(lat)
    if box :
        ds = ds.sel({lon:slice(box[0],box[1]),lat:slice(box[3],box[2])})
        weight = weight.sel({lon:slice(box[0],box[1]),lat:slice(box[3],box[2])})
        if mask is not None : 
            mask = mask.sel({lon:slice(box[0],box[1]),lat:slice(box[3],box[2])})
    if mask is None:
        mask=1
    
    mask = (mask*weight).where(~np.isnan(ds))
    ds   = mask*ds

    return ds.sum([lat,lon])/ mask.sum([lat,lon])

def ddx(F,lon='longitude'):
    """return zonal derivative in spherical coordinates"""
    coslat = np.cos(F.latitude*np.pi/180.)
    coslat += 1e-5*(1-1*(coslat>1e-5))
    m_per_degreelat = 6370*1e3*np.pi/180
    return F.differentiate(lon)/(m_per_degreelat*coslat)

def ddy(F,lat='latitude'):
    """return meridional derivative in spherical coordinates"""
    coslat = np.cos(F.latitude*np.pi/180.)
    coslat += 1e-5*(1-1*(coslat>1e-5))
    m_per_degreelat = 6370*1e3*np.pi/180
    return F.differentiate(lat)/m_per_degreelat

def divergence(Fx,Fy,lat='latitude',lon='longitude'):
    """return divergence in spherical coordinates"""
    coslat = np.cos(Fx[lat]*np.pi/180.)
    coslat += 1e-5*(1-1*(coslat>1e-5))
    m_per_degreelat = 6370*1e3*np.pi/180    
    return (Fx.differentiate(lon) + (Fy*coslat).differentiate(lat))/(m_per_degreelat*coslat)



###################################################################################################
#################################### BL & MISCELLANEOUS ###########################################
###################################################################################################

def crossslopeflow(u,v,angle):
    return (u*np.sin(angle*np.pi/180)+v*np.cos(angle*np.pi/180))

def compute_BL(thetaeB,thetaeL,thetaeLstar,wB=0.52):
    """Calculate the buoyancy surrogate B_L from thetaeB,thetaeL, and thetaeLstar.
    args:
        - thetaeB : array-like or float, boundary layer equivalent potential temperature in K
        - thetaeL : array-like or float, lower-free-tropospheric equivalent potential temperature in K
        - thetaeLstar : array-like or float, lower-free-tropospheric saturationequivalent potential 
          temperature in K
        - wB : importance of CAPE in setting plume buoyancy. See Ahmed, Adames & Neelin (JAS 2020).
    returns:
        - B_L, xr.DataArray, buoyancy in m/s^2
    """
    g=9.81
    wL = 1-wB
    thetae0 = 340
    capeL   = (thetaeB/thetaeLstar - 1)*thetae0
    subsatL = (1 - thetaeL/thetaeLstar)*thetae0
    BL = g/thetae0*(wB*capeL-wL*subsatL) # Did not include the KappaL term to conform with AAN2020
    return BL

###################################################################################################
##################################### STATISTICAL TOOLS ###########################################
###################################################################################################


def linregress_xr(x,y,dim='time'):
    """Linear regression of variable y on variable x along a specified dimension. 
    Wrote this vectorized function which is much faster than looping over a pre-defined function.
    args:
        - x : xr.DataArray, usually a timeseries 
        - y : xr.DataArray, can have any number of dimensions
        - dim : str, the dimension to perform the regression on
    returns:
        - xr.DataArray containing all the dimension in y except dim. Contains three variables:
        slope, rsquared, and pvalue, which chartacterize the linear regression at every point.
    """
    nt = len(x[dim])
    assert nt==len(y[dim])
    ssxm = nt*x.var(dim=dim)
    ssym = nt*y.var(dim=dim)
    ssxym = nt*xr.cov(x,y,dim=dim)       
    r = np.maximum(np.minimum(ssxym / np.sqrt(ssxm * ssym),1.),-1)
    slope = ssxym / ssxm
    
    df = nt - 2  # Number of degrees of freedom
    TINY = 1.0e-20
    t = r * np.sqrt(df / ((1.0 - r + TINY)*(1.0 + r + TINY)))
    
    pval = special.stdtr(df, -np.abs(t))*2 * y.isel({dim:0})**0
    
    return xr.merge([slope.rename('slope'),(r**2).rename('rsquared'),pval.rename('pvalue')])

def fdr(pvalues,alpha):
    """Given an array of p-values and a false discovery rate (FDR) level, 
    return the p-value threshold that defines significance according to this FDR level.
    See Wilks (2016) for more info.
    """
    sortidx = np.argsort(pvalues)
    psorted = pvalues[sortidx]
    psorted[np.isnan(psorted)]=1
    nval = len(pvalues)
    ifdr = np.argmax((psorted < alpha*np.arange(1,nval+1)/nval)[::-1])
    if ifdr == 0 and psorted[-1]>= alpha:
        ifdr=nval-1
    ifdr = nval - ifdr - 1
    return sortidx[:ifdr]

def fdr_xr_2d(pvalues,alpha):
    """Given an map of p-values and a false discovery rate (FDR) level, 
    return the mask that defines significance according to this FDR level.
    See Wilks (2016) for more info.
    """    
    pvalues=np.array(pvalues)
    assert len(pvalues.shape)==2
    ntot = pvalues.shape[0]*pvalues.shape[1]
    idxs_1d = fdr(pvalues.reshape(-1),alpha)
    flags = np.zeros(ntot)
    flags[idxs_1d] = 1
    return flags.reshape(pvalues.shape)*pvalues**0