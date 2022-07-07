#io for moist QG model
from netCDF4 import Dataset
import numpy as np

def create_file( filename, y, t ):
	ds = Dataset(filename, mode='w')

	ds.createDimension('time', size=t)
	ds.createDimension('y', size=y.shape[0])
	time = ds.createVariable('time', 'f4', dimensions=['time'])
	yn = ds.createVariable('y', 'f4', dimensions=['y'])
	yn.setncatts({'standard_name': 'y', 'units': 'degrees_east'})

	yn[:] = y
	time = np.linspace(0., float(t) - 1., t)

	zu1n = ds.createVariable(
		    'zu1',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	zu2n = ds.createVariable(
		    'zu2',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	ztaun = ds.createVariable(
		    'ztau',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	mn = ds.createVariable(
		    'zm',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	Pn = ds.createVariable(
		    'zP',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	En = ds.createVariable(
		    'zE',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	wn = ds.createVariable(
		    'zw',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	tauPn = ds.createVariable(
		    'ztauP',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	etauPn = ds.createVariable(
		    'etauP',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	wskewn = ds.createVariable(
		    'zwskew',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	eke1n = ds.createVariable(
		    'zeke1',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	eke2n = ds.createVariable(
		    'zeke2',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	emf1n = ds.createVariable(
		    'zemf1',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	emf2n = ds.createVariable(
		    'zemf2',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	ehf1n = ds.createVariable(
		    'zehf1',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)
	ehf2n = ds.createVariable(
		    'zehf2',
		    'f4',
		    dimensions=['time', 'y'],
		    zlib=True)

	zu1n.setncatts({'standard_name': 'zonal-mean u1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	zu2n.setncatts({'standard_name': 'zonal-mean u2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	ztaun.setncatts({'standard_name': 'zonal-mean tau',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	mn.setncatts({'standard_name': 'moisture',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	Pn.setncatts({'standard_name': 'precip',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	En.setncatts({'standard_name': 'evap',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	wn.setncatts({'standard_name': 'w',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	tauPn.setncatts({'standard_name': 'ztauP',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	etauPn.setncatts({'standard_name': 'etauP',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	wskewn.setncatts({'standard_name': 'w skew',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	eke1n.setncatts({'standard_name': 'eke 1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	eke2n.setncatts({'standard_name': 'eke 2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	emf1n.setncatts({'standard_name': 'emf 1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	emf2n.setncatts({'standard_name': 'emf 2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	ehf1n.setncatts({'standard_name': 'ehf 1',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})
	ehf2n.setncatts({'standard_name': 'ehf 2',
		                  'units': 'dimensionless',
		                  'grid_mapping': 'x_y'})

	return ds, zu1n, zu2n, ztaun, mn, Pn, En, wn, tauPn, etauPn, wskewn, eke1n, eke2n, emf1n, emf2n, ehf1n, ehf2n, time

def sdat(c, F):
    print("Saving in:", F)
    np.savez(F, u = c)
    return 0

def write_res_files( filename, psic1, psic2, qc1, qc2, mc, t0 ):
	
	sdat( psic1, filename + "_psic1.npz")
	sdat( psic2, filename + "_psic2.npz")
	sdat( qc1, filename + "_qc1.npz")
	sdat( qc2, filename + "_qc2.npz")
	sdat( mc, filename + "_mc.npz")
	sdat( t0, filename + "_t0.npz")

	return 0

def write_data_dry( ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2 ):
	
	ds.variables['zu1'][:] = zu1[:]
	ds.variables['zu2'][:] = zu2[:]
	ds.variables['ztau'][:] = ztau[:]
	ds.variables['zeke1'][:] = zeke1[:]
	ds.variables['zeke2'][:] = zeke2[:]
	ds.variables['zemf1'][:] = zemf1[:]
	ds.variables['zemf2'][:] = zemf2[:]
	ds.variables['zehf1'][:] = zehf1[:]
	ds.variables['zehf2'][:] = zehf2[:]
	ds.sync

	return 0

def write_data_moist( ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2, zm, zP, zE, zw, ztauP, zetauP, zwskew ):
	
	ds.variables['zu1'][:] = zu1[:]
	ds.variables['zu2'][:] = zu2[:]
	ds.variables['ztau'][:] = ztau[:]
	ds.variables['zeke1'][:] = zeke1[:]
	ds.variables['zeke2'][:] = zeke2[:]
	ds.variables['zemf1'][:] = zemf1[:]
	ds.variables['zemf2'][:] = zemf2[:]
	ds.variables['zehf1'][:] = zehf1[:]
	ds.variables['zehf2'][:] = zehf2[:]
	ds.variables['zm'][:] = zm[:]
	ds.variables['zP'][:] = zP[:]
	ds.variables['zE'][:] = zE[:]
	ds.variables['zw'][:] = zw[:]
	ds.variables['ztauP'][:] = ztauP[:]
	ds.variables['zetauP'][:] = zetauP[:]
	ds.variables['zwskew'][:] = zwskew[:]
	ds.sync

	return 0

def load_res_file( filename ):

    fpsic1 = np.load( filename + "_psic1.npz")
    psic1 = fpsic1['u'][:]
    fpsic2 = np.load( filename + "_psic2.npz")
    psic2 = fpsic2['u'][:]
    fqc1 = np.load( filename + "_qc1.npz")
    qc1 = fqc1['u'][:]
    fqc2 = np.load( filename + "_qc2.npz")
    qc2 = fqc2['u'][:]
    fmc = np.load( filename + "_mc.npz")
    mc = fmc['u'][:]
    ft0 = np.load( filename + "_t0.npz")
    t0 = ft0['u']

    return psic1, psic2, qc1, qc2, mc, t0

def load_dry_data( filename ):
    ds = Dataset(filename, mode='a')

    zu1 = ds.variables['zu1'][:]
    zu2 = ds.variables['zu2'][:]
    ztau = ds.variables['ztau'][:]
    zeke1 = ds.variables['zeke1'][:]
    zeke2 = ds.variables['zeke2'][:]
    zemf1 = ds.variables['zemf1'][:]
    zemf2 = ds.variables['zemf2'][:]
    zehf1 = ds.variables['zehf1'][:]
    zehf2 = ds.variables['zehf2'][:]
    time = ds.variables['time'][:]

    return ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2, time

def load_moist_data( filename ):
    ds = Dataset(filename, mode='a')

    zu1 = ds.variables['zu1'][:]
    zu2 = ds.variables['zu2'][:]
    ztau = ds.variables['ztau'][:]
    zeke1 = ds.variables['zeke1'][:]
    zeke2 = ds.variables['zeke2'][:]
    zemf1 = ds.variables['zemf1'][:]
    zemf2 = ds.variables['zemf2'][:]
    zehf1 = ds.variables['zehf1'][:]
    zehf2 = ds.variables['zehf2'][:]
    zm = ds.variables['zm'][:]
    zP = ds.variables['zP'][:]
    zE = ds.variables['zE'][:]
    zw = ds.variables['zw'][:]
    ztauP = ds.variables['ztauP'][:]
    zetauP = ds.variables['zetauP'][:]
    zwskew = ds.variables['zwskew'][:]
    time = ds.variables['time'][:]
    return ds, zu1, zu2, ztau, zeke1, zeke2, zemf1, zemf2, zehf1, zehf2, zm, zP, zE, zw, ztauP, zetauP, zwskew, time
