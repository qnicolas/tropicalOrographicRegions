import numpy as np
import xarray as xr

cp = 1004. # Heat capacity of air in J/kg/K
Lc = 2.5e6 # latent heat of condensation in J/kg
g = 9.81   # m/s2
Rv = 461   # Gas constant for water vapor, J/kg/K 
Rd = 287   # Gas constant for dry air, J/kg/K 

def k_vector(Nx,dx):
    """Given an x grid, return the grid of wavenumbers on which an FFT will be computed
    args:
     - Nx, number of points in grid
     - dx, grid spacing
    returns:
     - k, wavenumber array
    """
    return 2*np.pi*np.fft.fftfreq(Nx,dx)

def compute_Lq(Ms_ov_M,U,tauq):
    tauqtilde = 0.6*tauq # conversion from full-tropospheric average to lower tropospheric average, see Ahmed et al 2020
    return Ms_ov_M*U*tauqtilde

def lapse_rates():
    """Compute the lapse rates and B-V frequency for input into the linear theory
    Assumptions : 
     * ds0/dz=3K/km
     * dq0dz computed from an exponentially decreasing profile starting from 80%RH at 300K and moisture scale height = 2.5km (=-8.1K/km), averages from 1000m to 3000 m"""
    zbot=1000
    ztop=3000

    ds0dz = cp * 3e-3 # 3K/km
    dq0dz = -Lc * 0.8 *0.022/2500 * 2500/(ztop-zbot) * (np.exp(-zbot/2500)-np.exp(-ztop/2500)) # 0.022 = qsat at 300K, used 80%RH and moisture scale height = 2.5km
    N=np.sqrt(g/300 * ds0dz/cp)

    return ds0dz,dq0dz,N


def m_exponent_2D(sigma,N,ksq):
    """Vectorized function to compute the vertical wavenumbers in linear mountain wave theory
    args:
     - sigma : array-like, wavenumber in the cross-slope direction (U*k_x+V*k_y)
     - N : float, Brunt-Vaisala frequency
     - U : float, basic-state wind
    returns:
     - m : array-like, vertical wavenumber
    """
    den = sigma**2
    EPS=1e-15
    den[den < EPS] = EPS
    return (den>=N**2)*1j*np.sqrt(ksq*(1-N**2/den)+0.j) + (den<N**2)*np.sign(sigma)*np.sqrt(ksq*(N**2/den-1)+0.j)



########################################################################################
############################  NICOLAS & BOOS (2022) THEORY  ############################
########################################################################################

def linear_precip_theory_2D(xx,yy,hxy,U,V,N,tauT=7.5,tauq=27.5,P0=4.,switch=1,pad_factor=0.2):
    """Computes the precipitation profile predicted by the linear theory (equation (17) in Nicolas&Boos 2023).
    Assumptions : 
     * assumptions about lapse rates detailed in the function lapse_rates
     * Averages taken between z=1000m and z=3000m
     * Normalized gross moist stability = 0.2
     * pT/g=8000 kg/m2
    args:
     - xx : array-like, x-grid in m.
     - yy : array-like, y-grid in m.
     - hxy : array-like, topographic profile in m
     - U : float, x component of basic-state wind in m/s
     - V : float, y component of basic-state wind in m/s
     - N : float, Brunt-Vaisala frequency
     - tauT : float, temperature adjustment time scale in hours
     - tauq : float, moisture adjustment time scale in hours
     - P0 : float, basic-state precipitation in mm/day
     - switch : int, set to 0 to turn off the effect of Lq (equivalent to setting Lq=0, no precip relaxation).
     - pad_factor : float, fraction of initial domain width used to pad the topography to 0 on the edges
       Note topography is linearly brought to 0 on a 100 km distance
    returns:
     - P : array_like, precipitation in mm/day
     """
    pT_ov_g = 8e3 #mass of troposphere in kg/m2
    Lc=2.5e6;g=9.81;cp=1004.
    dx = xx[1]-xx[0]
    dy = yy[1]-yy[0]
    tauT*=3600
    tauq*=3600
    
    # Pad boundaries
    calc_pad = int(pad_factor*np.max(hxy.shape))
    pad=calc_pad#pad = min([calc_pad, 200])
    pad_topo = int(100e3/dx)
    hxy_pad_topo = np.pad(hxy, pad_topo, 'linear_ramp',end_values = [0,0])
    hxy_pad = np.pad(hxy_pad_topo,pad-pad_topo,'constant')
    xx_pad = np.pad(xx, pad, 'linear_ramp',end_values = [xx[0]-pad*dx,xx[-1]+pad*dx])
    yy_pad = np.pad(yy, pad, 'linear_ramp',end_values = [yy[0]-pad*dy,yy[-1]+pad*dy])
    
    z=np.arange(0,10000,100)
    kx=k_vector(len(xx_pad),dx)
    ky=k_vector(len(yy_pad),dy)
    sigma = U*kx[:,None]+V*ky[None,:]
    ksq = kx[:,None]**2+ky[None,:]**2
    
    LqovU=compute_Lq(5,1,tauq)

    ds0dz = cp*300.*N**2/g
    _,dq0dz,_ = lapse_rates()
    chi = pT_ov_g * (ds0dz/tauT - dq0dz/tauq)/ Lc * 86400
    
    zbot=1000
    ztop=3000    
    z_slice = z[np.where((z>=zbot) & (z<=ztop))]
    
    
    m1 = m_exponent_2D(sigma,N,ksq)
    mm = np.copy(m1)
    mm[mm==0]=1e-8
    
    Pprimehat = (1j*sigma/(1j*sigma + switch*1/LqovU)) * chi * np.fft.fft2(hxy_pad) * ((m1!=0)*(np.exp( 1j* mm * ztop )-np.exp( 1j* mm * zbot ))/(1j*mm*(ztop-zbot)) + (m1==0)*1) 
    # equivalently np.exp( 1j* m_exponent_2D(sigma,N,ksq)[:,:,None] *  z_slice[None,None,:] ).mean(axis=-1) ?

    P = P0 + np.real(np.fft.ifft2(Pprimehat))
    P = np.maximum(0.,P)[pad:-pad, pad:-pad]
    return xr.DataArray(P,coords={'x':xx,'y':yy},dims=['x','y'])

    

def p_lineartheory_region(MR,topo='ETOPO',N=0.01,tauT=7.5,tauq=27.5,pad='small',switch=1):
    """Computes the precipitation profile predicted by the linear theory (equation (17) in Nicolas&Boos 2023),
    for a given mountain range object.
    args:
     - MR : MountainRange, must contain topography and model parameters
     - topo : str, description of topography data to use ('ETOPO' for ETOPO data, 'ETOPOCOARSE' for 
       ETOPO coarsened by a factor 4, or 'ERA5' for ERA5 topography)
     - N : float, Brunt-Vaisala frequency
     - tauT : float, temperature adjustment time scale in hours
     - tauq : float, moisture adjustment time scale in hours
     - pad : str, 'small' or 'big', determines the fraction of initial domain width used to pad the topography 
       to 0 on the edges.
     - switch : int, set to 0 to turn off the effect of Lq (equivalent to setting Lq=0, no precip relaxation).
    returns : 
     - P : xr.DataArray, precipitation, defined on the same grid as the topography
    """
    if topo=='ETOPO':
        z = MR.vars['Z_HR']
    elif topo=='ETOPOCOARSE':
        z = MR.vars['Z_HR'].coarsen(latitude=4,longitude=4,boundary='trim').mean()
    elif topo=='ERA5':
        z = MR.vars['Z']
    else:
        raise ValueError('topo')
    hxy = np.array(z).T[:,::-1]
    lon = z.longitude
    lat = z.latitude[::-1]
    
    xx = np.array(lon)*100e3
    yy = np.array(lat)*100e3
    
    if pad=='small':
        pf=0.2
    elif pad=='big':
        pf=2
    P = linear_precip_theory_2D(xx,yy,hxy,MR.U0,MR.V0,N,tauT=tauT,tauq=tauq,P0=MR.P0,pad_factor=pf,switch=switch)
    return P.assign_coords({'longitude':P.x/100e3,'latitude':P.y/100e3}).swap_dims({'x':'longitude','y':'latitude'})[:,::-1].transpose()





########################################################################################
###########################  SMITH & BARSTAD (2004) THEORY  ############################
########################################################################################

def humidsat(t,p):
    """computes saturation vapor pressure (esat), saturation specific humidity (qsat),
    and saturation mixing ratio (rsat) given inputs temperature (t) in K and
    pressure (p) in hPa.
    
    these are all computed using the modified Tetens-like formulae given by
    Buck (1981, J. Appl. Meteorol.)
    for vapor pressure over liquid water at temperatures over 0 C, and for
    vapor pressure over ice at temperatures below -23 C, and a quadratic
    polynomial interpolation for intermediate temperatures."""
    
    tc=t-273.16
    tice=-23
    t0=0
    Rd=287.04
    Rv=461.5
    epsilon=Rd/Rv


    # first compute saturation vapor pressure over water
    ewat=(1.0007+(3.46e-6*p))*6.1121*np.exp(17.502*tc/(240.97+tc))
    eice=(1.0003+(4.18e-6*p))*6.1115*np.exp(22.452*tc/(272.55+tc))
    #alternatively don't use enhancement factor for non-ideal gas correction
    #ewat=6.1121*exp(17.502*tc/(240.97+tc));
    #eice=6.1115*exp(22.452*tc/(272.55+tc));
    eint=eice+(ewat-eice)*((tc-tice)/(t0-tice))*((tc-tice)/(t0-tice))

    esat=(tc<tice)*eice + (tc>t0)*ewat + (tc>tice)*(tc<t0)*eint

    #now convert vapor pressure to specific humidity and mixing ratio
    rsat=epsilon*esat/(p-esat);
    qsat=epsilon*esat/(p-esat*(1-epsilon));
    
    return esat,qsat,rsat

def hw_cw(ts,ps,gamma,gamma_m):
    """Compute water vapor scale height and coefficient Cw for the Smith&Barstad (2004) model.
    args:
     - ts, surface temperature in K
     - ps, surface pressure in Pa
     - gamma, environmental lapse rate in K/m
     - gamma_m, moist-adiabatic lapse rate in K/m
    returns:
     - Hw, Water vapor scale height in m
     - Cw, Uplift sensitivity factor in kg/m^3
    """

    Hw = Rv*ts**2/(Lc*gamma)
    Cw = humidsat(ts,ps/100)[0]*100/Rd/ts*gamma_m/gamma
    return Hw,Cw

def smith_theory_2D(xx,yy,hxy,U,V,N,gamma_m,ts=300.,ps=100000.,tau=1000, P0=4.,pad_factor=0.2):
    """Compute precipitation from the Smith&Barstad (2004) model. 
    Because in the tropics the environmental lapse rate is steeper than the moist adiabat (conditionally unstable 
    environment), a dry static stability is used to compute airflow dynamics.
    args:
     - xx : array-like, x-grid in m.
     - yy : array-like, y-grid in m.
     - hxy : array-like, topographic profile in m
     - U : float, x component of basic-state wind in m/s
     - V : float, y component of basic-state wind in m/s
     - N : float, Brunt-Vaisala frequency
     - gamma_m : float, moist-adiabatic lapse rate in K/m
     - ts : float, surface temperature in K
     - ps : float, surface pressure in Pa
     - tau : float, conversion and fallout time scale in s
     - P0 : float, basic-state precipitation in mm/day
     - pad_factor : float, fraction of initial domain width used to pad the topography to 0 on the edges
       Note topography is linearly brought to 0 on a 100 km distance
    returns:
     - P : array_like, precipitation in mm/day
    """
    dx = xx[1]-xx[0]
    dy = yy[1]-yy[0]
    g=9.81;cp=1004.
    gamma = g/cp - ts*N**2/g
    Hw,Cw = hw_cw(ts,ps,gamma,gamma_m)
    
    tau_c=tau
    tau_f=tau
    
    # Pad boundaries
    calc_pad = int(pad_factor*np.max(hxy.shape))
    pad=calc_pad#pad = min([calc_pad, 200])
    pad_topo = int(100e3/dx)
    hxy_pad_topo = np.pad(hxy, pad_topo, 'linear_ramp',end_values = [0,0])
    hxy_pad = np.pad(hxy_pad_topo,pad-pad_topo,'constant')
    xx_pad = np.pad(xx, pad, 'linear_ramp',end_values = [xx[0]-pad*dx,xx[-1]+pad*dx])
    yy_pad = np.pad(yy, pad, 'linear_ramp',end_values = [yy[0]-pad*dy,yy[-1]+pad*dy])
    
    z=np.arange(0,10000,100)
    kx=k_vector(len(xx_pad),dx)
    ky=k_vector(len(yy_pad),dy)
    sigma = U*kx[:,None]+V*ky[None,:]
    ksq = kx[:,None]**2+ky[None,:]**2
    
    m1 = m_exponent_2D(sigma,N,ksq)
    mm = np.copy(m1)
    mm[mm==0]=1e-8
    Pprimehat= 86400*Cw*np.fft.fft2(hxy_pad)*1j*sigma/(1-1j*Hw*mm)/(1+1j*sigma*tau_c)/(1+1j*sigma*tau_f)/2.5

    P = P0 + np.real(np.fft.ifft2(Pprimehat))
    P = np.maximum(0.,P)[pad:-pad, pad:-pad]
    return xr.DataArray(P,coords={'x':xx,'y':yy},dims=['x','y'])
    

def p_smiththeory_region(MR,topo='ETOPO',N=0.01,gamma_m=4.32e-3,tau=1000,pad='small'):
    """Computes the precipitation profile from the Smith&Barstad (2004) model. 
    args:
     - MR : MountainRange, must contain topography and model parameters
     - topo : str, description of topography data to use ('ETOPO' for ETOPO data, 'ETOPOCOARSE' for 
       ETOPO coarsened by a factor 4, or 'ERA5' for ERA5 topography)
     - N : float, Brunt-Vaisala frequency
     - gamma_m : float, moist-adiabatic lapse rate in K/m
     - tau : float, conversion and fallout time scale in s
     - pad : str, 'small' or 'big', determines the fraction of initial domain width used to pad the topography 
       to 0 on the edges.
    returns : 
     - P : xr.DataArray, precipitation, defined on the same grid as the topography
    """
    if topo=='ETOPO':
        z = MR.vars['Z_HR']
    elif topo=='ETOPOCOARSE':
        z = MR.vars['Z_HR'].coarsen(latitude=4,longitude=4,boundary='trim').mean()
    elif topo=='ERA5':
        z = MR.vars['Z']
    else:
        raise ValueError('topo')
    hxy = np.array(z).T[:,::-1]
    lon = z.longitude
    lat = z.latitude[::-1]
    
    xx = np.array(lon)*100e3
    yy = np.array(lat)*100e3
    
    if pad=='small':
        pf=0.2
    elif pad=='big':
        pf=2
    
    P = smith_theory_2D(xx,yy,hxy,MR.U0,MR.V0,N,gamma_m,tau=tau,P0=MR.P0,pad_factor=pf)
    return P.assign_coords({'longitude':P.x/100e3,'latitude':P.y/100e3}).swap_dims({'x':'longitude','y':'latitude'})[:,::-1].transpose()


########################################################################################
#################################  UPSLOPE THEORY  #####################################
########################################################################################

def p_upslope_region(MR,topo='ERA5',version='IVT'):
    """Computes the precipitation profile from a simple upslope model. 
    args:
     - MR : MountainRange, must contain topography and model parameters
     - topo : str, description of topography data to use ('ETOPO' for ETOPO data, 'ETOPOCOARSE' for 
       ETOPO coarsened by a factor 4, or 'ERA5' for ERA5 topography)
     - version : str, whether to use IVT or surface specific humidity times wind
    returns : 
     - P : xr.DataArray, precipitation, defined on the same grid as the topography
    """
    if topo=='ETOPO':
        z = MR.vars['Z_HR']
    elif topo=='ETOPOCOARSE':
        z = MR.vars['Z_HR'].coarsen(latitude=4,longitude=4,boundary='trim').mean()
    elif topo=='ERA5':
        z = MR.vars['Z']
    else:
        raise ValueError('topo')
    m_per_degreelat = 6370*1e3*np.pi/180
    rho0=1.2 # surface air density in kg/m^3
    Hm=2500. # moisture scale height in m
    if version=='surf':
        p = 86400*rho0 * MR.vars['Q_SFC'] * (MR.vars['VAR_100U']*z.differentiate('longitude')/m_per_degreelat
                                        +MR.vars['VAR_100V']*z.differentiate('latitude')/m_per_degreelat)
    elif version=='IVT':
        p = 86400* (MR.vars['VIWVE']*z.differentiate('longitude')/m_per_degreelat
                   +MR.vars['VIWVN']*z.differentiate('latitude')/m_per_degreelat) / Hm 
    
    return np.maximum(p,0.)

