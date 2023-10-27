import os
import numpy as np
import xarray as xr
from scipy.ndimage import rotate
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size':15})
import time
import cartopy.crs as ccrs
import cartopy

from tools import *

def rotate_var(var,angle,two_dim=True,**rotate_args):
    """Given a variable on a (lat,lon) grid and an angle (in degrees), rotate the variable.
    args :
        - var : xr.Dataarray, array to be rotated
        - angle : float, angle that defines the new x axis after array rotation (in degrees). 
                  If the angle is 90° (i.e. pointing East), the array is not rotated.
        - two_dim : bool, whether the array has two dimensions (True) or more (False)
                    if True, the dimensions must be order so that (lat,lon) are the last two ones
    returns :
        - ds : xarray.Dataarray, with dimensions x and y, where x is along the axis 
               defined by the angle (e.g. x goes west to east if angle=90, or southwest 
               to northeast if angle=45)
    """
    dx = 2.*np.pi*6400/360*(var.longitude[1]-var.longitude[0])# approximate everything as at the equator
    n=len(var.longitude)
    m=len(var.latitude)
    x = np.arange(0,dx*(n+1),dx)
    y = np.arange(0,dx*m,dx)
    
    if two_dim:
        var_rot_ar = rotate(np.array(var)[::-1],90-angle,reshape=False,cval=np.nan,**rotate_args)
        return xr.DataArray(var_rot_ar,coords={'x':x[:var_rot_ar.shape[-1]],'y':y[:var_rot_ar.shape[0]]},dims=['y','x'])
    else:
        var_rot_ar = rotate(np.array(var)[...,::-1,:],90-angle,axes=(-1,-2),reshape=False,cval=np.nan,**rotate_args)
        coords = dict(var.coords).copy()
        if "latitude" in coords: 
            del coords['latitude']
        if "longitude" in coords: 
            del coords['longitude']
        return xr.DataArray(var_rot_ar,coords={'x':x[:var_rot_ar.shape[-1]],'y':y,**coords},dims=[*var.dims[:-2],'y','x'])
    
def tilted_rect(grid,x1,y1,x2,y2,width,reverse=False):
    """Creates a mask of a box that has a tilted rectangular shape (to outline rain bands)
    args :
        - grid : xr.Dataarray, defines the grid over which the mask will be created
        - x1 : float, longitude of the bottom-left corner of the rectangle
        - y1 : float, latitude of the bottom-left corner of the rectangle
        - x2 : float, longitude of the top-left corner of the rectangle
        - y2 : float, latitude of the top-left corner of the rectangle
        - width : float, width of the rectangle
        - reverse : bool, whether to flip the rectangle with respects to the ((x1,y1),(x2,y2)) axis
    returns :
        - xr.Dataarray, field of 0s and 1s defined on the grid that outline the tilted rectangle
    """
    x = grid.longitude
    y = grid.latitude
    if reverse:
        halfplane_para = (x-x1)*(y2-y1) - (x2-x1)*(y-y1) <=0
    else:
        halfplane_para = (x-x1)*(y2-y1) - (x2-x1)*(y-y1) >=0
    sc_prod = (x-x1)*(x2-x1)+(y-y1)*(y2-y1)
    halfplane_perp_up = sc_prod >= 0
    halfplane_perp_dn = (x-x2)*(x1-x2)+(y-y2)*(y1-y2) >= 0
    distance_across = np.sqrt(np.maximum(0.,(x-x1)**2+(y-y1)**2 - sc_prod**2/((x2-x1)**2+(y2-y1)**2)))
    return (halfplane_para*halfplane_perp_up*halfplane_perp_dn*(distance_across<width)).transpose('latitude','longitude')

def tilted_rect_distance(grid,x1,y1,x2,y2,distups1,distups2):
    """Creates a mask of a box that has a tilted rectangular shape (to outline rain bands or regions upstream of them)
    This is a more general version of the above function. The idea is that we define an axis with two points. The rectangle
    will have two sides parallel to this axis, with length the distance between these two points. Their location and the 
    distance between them are defined by distups1 and distups2, which define distances upstream of the reference axis, 
    perpendicular to this axis.
    args :
        - grid : xr.Dataarray, defines the grid over which the mask will be created
        - x1 : float, longitude of the first point defining the reference axis
        - y1 : float, latitude of the first point defining the reference axis
        - x2 : float, longitude of the second point defining the reference axis
        - y2 : float, latitude of the second point defining the reference axis
        - distups1 : distance of the first side along of the reference axis. Negative distances allowed.
        - distups2 : distance of the second side along of the reference axis. Must be < distups1
    returns :
        - xr.Dataarray, field of 0s and 1s defined on the grid that outline the tilted rectangle
    """
    x = grid.longitude
    y = grid.latitude
    scalarprod_para = (x-x1)*(y2-y1) - (x2-x1)*(y-y1)
    scalarprod_perp_up = (x-x1)*(x2-x1)+(y-y1)*(y2-y1)
    scalarprod_perp_dn = (x-x2)*(x1-x2)+(y-y2)*(y1-y2)
    halfplane_para = scalarprod_para >=0
    halfplane_perp_up = scalarprod_perp_up >= 0
    halfplane_perp_dn = scalarprod_perp_dn >= 0
    distance_across = np.sqrt((x-x1)**2+(y-y1)**2 - scalarprod_perp_up**2/((x2-x1)**2+(y2-y1)**2))*np.sign(scalarprod_para)
    return (halfplane_perp_up*halfplane_perp_dn*(distance_across<distups1)*(distance_across>distups2)).transpose('latitude','longitude')


class MountainRange :
    """
    This class is used to hold all data and attributes of a given region, as well as compute various diagnostics
    """
    def __init__(self, name, box, Lname, angle, months, box_tilted, path= '/pscratch/sd/q/qnicolas/regionsDataBig/'):
        """Initialize a MountainRange object.
        args :
            - name       : str, short name that defines the region (used in all data file names, etc.)
            - box        : list of 4 float [lon1, lon2, lat1, lat2] where lon1<lon2 and lat1<lat2, defines 
                           region extent
            - Lname      : str, long name describing the region (used on plots)
            - angle      : float, cross-slope angle. It defines the direction along which all cross-sections will be
                           taken.
            - months     : list of int, defines the season over which all analyses are made (e.g. [6,7,8] for JJA)
            - box_tilted : list of 5 float, defines the box that outlines the rain band region
                           template is: [lon1, lat1, lon2, lat2, width] where (lon1, lat1) define the bottom-right 
                           corner, (lon2, lat2) define the top-right corner, and width the width of the box in °.
            - path       : str, path where all data related to this region are stored  
        """
        self.name=name
        self.box=box
        self.Lname=Lname
        self.angle = angle
        self.months=months
        self.vars={}
        self.path=path
        self.box_tilted = box_tilted
        self.years = range(2001,2021)
        self._monthstr = '-'.join(["{:02}".format(m) for m in self.months])
        
    def subset_2dvar(self,varname,var):
        """Select a variable in the region's box and season, and place in the vars dictionnary
        args :
            - varname : str, key to access the variable in the vars dictionnary
            - var     : xr.Dataarray
        """
        self.vars[varname] = sel_box_months(var,self.box,self.months)

    def set_uperp(self):
        """Calculate the cross-slope velocity component. Requires VAR_100U and VAR_100V
        to be loaded"""
        self.vars['VAR_100U_PERP'] = crossslopeflow(self.vars['VAR_100U'], self.vars['VAR_100V'],self.angle)
        
    def set_viwvperp(self):
        """Calculate the cross-slope IVT component. Requires VIWVE_DAILY and 
        VIWVN_DAILY to be loaded"""
        self.vars['VIWV_PERP_DAILY'] = crossslopeflow(self.vars['VIWVE_DAILY'], self.vars['VIWVN_DAILY'],self.angle)

    def set_era5_var(self,varcode,varname,group='sfc'):
        """Fetch an ERA5 variable that has already been subset for this region.
        args :
            - varcode : str, ERA5 variable code, e.g. 128_034_sstk
            - varname : str, key to access the variable in the vars dictionnary
            - group : str, ERA5 variable type, e.g. 'sfc', 'vinteg', or 'pl'
        """
        filepath = self.path+"e5.oper.an.{}.{}.ll025sc.{}-{}.{}.{}.nc".format(group,varcode,self.years[0],self.years[-1],self._monthstr,self.name)
        self.vars[varname] = xr.open_dataarray(filepath)
        
    def set_spatialmean(self,varname,locname,mask,box=None):
        """Calculate the spatial mean of a variable inside a given mask and box.
        args :
            - varname : str, key to access the variable in the vars dictionnary
            - locname : str, ERA5 variable type, e.g. 'sfc', 'vinteg', or 'pl'
            - mask : xr.Dataarray, defines the mask over which to average (spatial field of 0s and 1s)
            - box : optional, list of 4 int that define a rectangular box over which to average
        """
        self.vars[varname +'_'+ locname.upper()] = spatial_mean(self.vars[varname],box=box,mask=mask)
                
    def set_daily_imerg(self):
        """Fetch IMERG data for this region, at daily resolution."""
        filepath = self.path+"gpm_imerg_v06.{}-{}.{}.{}.nc".format(self.years[0],self.years[-1],self._monthstr,self.name)
        self.vars['IMERG_DAILY'] = xr.open_dataarray(filepath)
          
    def set_daily_Bl_vars(self,include_thetaeL=False):
        """Fetch boundary-layer and lower-free-tropospheric averaged thermodynamic quantities:
        T_L, q_L, thetaeb, thetaeL, thetaeLstar + calculate BL
        args :
            - include_thetaeL : whether to store THETAEL_DAILY and THETAELSTAR_DAILY in the vars dictionnary
        """
        wB = 0.52
        filepaths = [self.path+"e5.diagnostic.{}.{}-{}.{}.{}.nc".format(varcode,self.years[0],self.years[-1],self._monthstr,self.name) for varcode in ("thetaeb", "thetaeL", "thetaeLstar", "tL", "qL")]
        thetaeb = xr.open_dataarray(filepaths[0])
        thetaeL = xr.open_dataarray(filepaths[1])
        thetaeLstar = xr.open_dataarray(filepaths[2])
        self.vars['THETAEB_DAILY'] = thetaeb
        BL = compute_BL(thetaeb,thetaeL,thetaeLstar,wB)
        self.vars['BL_DAILY'] = BL
        tL = xr.open_dataarray(filepaths[3])
        qL = xr.open_dataarray(filepaths[4])
        self.vars['TL_DAILY'] = tL
        self.vars['QL_DAILY'] = qL
        if include_thetaeL:
            self.vars['THETAEL_DAILY'] = thetaeL
            self.vars['THETAELSTAR_DAILY'] = thetaeLstar

