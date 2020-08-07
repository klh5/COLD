import datacube
from datacube.utils import geometry
from datacube.virtual import construct_from_yaml
import geopandas as gpd
import json
import os
import csv
from datacube.storage.masking import mask_invalid_data
import numpy as np
import pandas as pd
from cold import transformToArray, runCOLD, datesToNumbers
from multiprocessing import Pool
from sklearn.cluster import KMeans
import datetime
from osgeo import gdal
import subprocess
import re
import xarray as xr
from multiprocessing import Manager
from sklearn.linear_model import LinearRegression
import salem
import argparse
import fcntl
from scipy.stats.distributions import chi2

# Data cube product recipe
combined_ls_sref = construct_from_yaml("""
    collate:
        - product: ls4_arcsi_sref_global_mangroves
          measurements: [red, green, NIR, SWIR1, SWIR2]
        - product: ls5_arcsi_sref_global_mangroves   
          measurements: [red, green, NIR, SWIR1, SWIR2]
        - product: ls7_arcsi_sref_global_mangroves   
          measurements: [red, green, NIR, SWIR1, SWIR2]
        - product: ls8_arcsi_sref_global_mangroves   
          measurements: [red, green, NIR, SWIR1, SWIR2]
    """)

def getDataset(time, poly, crs):
    
    fetch_ds = combined_ls_sref.query(dc, geopolygon=poly, time=time, resolution=(-30, 30), output_crs='EPSG:{}'.format(crs))

    grouped_ds = combined_ls_sref.group(fetch_ds)
            
    ds = combined_ls_sref.fetch(grouped_ds)    
            
    ds = ds.sortby('time')

    ds = mask_invalid_data(ds)

    ds = ds.dropna('time', how='all')
    
    return(ds)
    
def getModelCoeffs(x, y):
    
    # Get slope and intercept of maximum NDVI values
    
    global results
    
    px = all_ndvi.sel(x=x, y=y).NDVI
    
    lm = LinearRegression(fit_intercept=True).fit(px.year.values.reshape(30, 1), px.values)
    
    slope = lm.coef_[0]
    intercept = lm.intercept_
    
    results.append({'x': x, 'y': y, 'slope': slope, 'intercept': intercept})
    
def purge(dir, pattern):
    
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))
    
def runChangeDetection(x, y):

    input_ts = ds.sel(x=x, y=y)
    
    input_ts = transformToArray(input_ts)
    
    runCOLD(input_ts, bands, output_file, args.use_temporal, args.re_init, ch_thresh, args.alpha, x=x, y=y)

def writeOutPixel(x, y):  
    
    all_rows = []
    
    for r in rows:
        new_row = [x, y]
        new_row.extend(r)
        all_rows.append(new_row)
    
    with open(output_file, 'a') as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        writer = csv.writer(output)
        writer.writerows(all_rows)   
        fcntl.flock(output, fcntl.LOCK_UN)
    
parser = argparse.ArgumentParser(description="Run CCDC algorithm using Data Cube.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--outfile', default="output.csv", help="The output file.")
parser.add_argument('-s', '--use_spatial', default=False, action='store_true', help="Whether to use spatial clustering to speed up computation. False if not specified.")
parser.add_argument('-t', '--use_temporal', default=False, action='store_true', help="Whether to use temporal jumps to speed up computation. False if not specified.")
parser.add_argument('-g', '--gridfile', type=int, default=None, help="Shapefile containing gridded polygon of the area.")
parser.add_argument('-i', '--tile', type=int, default=None, help="The tile to process.")
parser.add_argument('-N', '--infile', default=None, help="The NetCDF file to process.")
parser.add_argument('-r', '--re_init', type=int, default=0, help="Days to wait before re-initializing the model.")
parser.add_argument('-a', '--alpha', type=float, default=1, help="Alpha value for Lasso fitting.")
parser.add_argument('-k', '--k_clusters', type=int, default=5, help="Number of clusters for k-means.")
parser.add_argument('-M', '--min_area', type=int, default=360000, help="Minimum area for clusters.")
parser.add_argument('-B', '--buffer', type=int, default=150, help="Buffer value for reducing cluster edges.")
parser.add_argument('-p', '--processes', type=int, default=4, help="Number of processes to use.")
parser.add_argument('-E', '--epsg', default=None, help="EPSG code if product doesn't specify (number only e.g. 4326).")

args = parser.parse_args()

tile = args.tile

print("Loading data...")

if(tile):
    
    grd_path = args.gridfile
        
    grd = gpd.read_file(grd_path)
    
    dc = datacube.Datacube()
        
    curr_poly = grd.where(grd.id == tile).dropna().iloc[0].geometry
    
    json_poly = json.loads(gpd.GeoSeries([curr_poly]).to_json())
    
    dc_geom = geometry.Geometry(json_poly['features'][0]['geometry'], geometry.CRS("EPSG:{}".format(args.epsg)))
    
    ds = getDataset(('1988-01-01', '2020-12-31'), dc_geom, args.epsg)
    
    dc.close()
    
else: 
    
    ds = xr.open_dataset(args.infile)

bands = list(ds.data_vars)

# Change threshold based on chi square distribution
ch_thresh = chi2.ppf(0.99, df=len(bands))

all_coords = [(x, y) for x in ds.x.values for y in ds.y.values]

headers = ["x", "y", "band", "start_date", "end_date", "start_val", "end_val", "coeffs", "RMSE", "intercept", "change_date", "magnitude"]

dirsplit = args.outfile.rsplit('/', 1)   

output_dir = dirsplit[0]

if(tile):
    filename = "{}_{}".format(tile, dirsplit[1])

else:
    input_name = args.infile.split('/')[-1].split('.')[0]
    filename = "{}_{}".format(input_name, dirsplit[1])
    tile = filename[0]
    
output_file = "{}/{}".format(dirsplit[0], filename)    
    
with open(output_file, 'w+') as output:
    writer = csv.writer(output)
    writer.writerow(headers)

if(args.use_spatial):

    datasets = []
    
    years = list(np.unique(pd.to_datetime(ds.time.values).year))
    
    for year in years:
        
        print(year)
        
        ndvi_ds = ds.sel(time=str(year))
        
        if not 'NDVI' in ds.data_vars:
            ndvi_ds['NDVI'] = (ndvi_ds.NIR - ndvi_ds.red) / (ndvi_ds.NIR + ndvi_ds.red)
            ndvi_ds = ndvi_ds.max(dim='time')
        else:
            ndvi_ds = ndvi_ds.max(dim='time')

        datasets.append(ndvi_ds)
    
    all_ndvi = xr.concat(datasets, dim='year')
    
    print("Data loaded")
    
    print("Fetching coefficients...")
    
    manager = Manager()
    results = manager.list()
                                  
    with Pool(processes=args.processes) as pool:
        pool.starmap(getModelCoeffs, all_coords)
    
    results = results._getvalue()
    
    print("Done")
    
    df_res = pd.DataFrame(results)
    
    samples = df_res.drop(['x', 'y'], axis=1)
    
    print("Running clustering...")
    
    kmeans = KMeans(n_clusters=args.k_clusters, init='k-means++', max_iter=50, n_init=10)        
    
    pred = kmeans.fit_predict(samples)
    
    print("Done")
    
    df_res['pred'] = pred
    
    df_res['pred'] += 1
    
    df_res = df_res[['x', 'y', 'pred']]
    
    output = df_res.set_index(['y', 'x']).to_xarray()     
    
    output = output.sortby("y", ascending=False) 
    
    output = output.fillna(0)
    
    print("Generating output file...")
    
    kmeans_out = output_dir + "/kmeans_out_{}.kea".format(tile)
    
    # Output to KEA file
    x_size = len(output.x.values)
    y_size = len(output.y.values)
    x_min = np.amin(output.x.values)
    y_max = np.amax(output.y.values)
    
    geo_transform = (x_min, 30, 0.0, y_max, 0.0, -30)
    
    driver = gdal.GetDriverByName('KEA')
    output_raster = driver.Create(kmeans_out, x_size, y_size, 1, 1) # Only one band, byte data type since there are only 5 values
    
    output_raster.SetProjection('PROJCS["Gulshan 303 / Bangladesh Transverse Mercator",GEOGCS["Gulshan 303",DATUM["Gulshan_303",SPHEROID["Everest 1830 (1937 Adjustment)",6377276.345,300.8017,AUTHORITY["EPSG","7015"]],TOWGS84[283.7,735.9,261.1,0,0,0,0],AUTHORITY["EPSG","6682"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4682"]],PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],PARAMETER["central_meridian",90],PARAMETER["scale_factor",0.9996],PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["Easting",EAST],AXIS["Northing",NORTH],AUTHORITY["EPSG","3106"]]')
    output_raster.SetGeoTransform(geo_transform)
    
    raster_band = output_raster.GetRasterBand(1)
    raster_band.SetNoDataValue(0)
    raster_band.SetDescription("cluster")
    
    raster_band.WriteArray(output.pred.values)
        
    output_raster = None
    
    kmeans_poly = output_dir + "/clusters_{}.shp".format(tile)
    
    print("Polygonizing...")
    
    # Polygonize the clusters given by k-means
    cmd = ["gdal_polygonize.py", "-q", kmeans_out, kmeans_poly]
         
    run_cmd = subprocess.run(cmd)
    
    # Read in the polygonized file
    clusters = gpd.read_file(kmeans_poly)
    
    print("Screening clusters...")
    
    # Process the clusters
    clusters = clusters[clusters.area >= args.min_area] # Only keep large polygons 
    
    clusters = clusters.buffer(-args.buffer) # Shrink all polygons to avoid edges
    
    clusters = clusters[~clusters.is_empty]
    
    # Explode polygons so that multipart clusters get separated
    clusters = clusters.explode()
    
    clusters = clusters.reset_index(drop=True)
    
    final_clusters_file = output_dir + '/final_clusters_{}.shp'.format(tile)
    
    clusters.to_file(final_clusters_file)
    
    num_clusters = len(clusters)
    
    print("Found {} clusters".format(num_clusters))
    
    #purge("/home/a.klh5/ccdc_spacetime/", "^clusters.*")
    #purge("/home/a.klh5/ccdc_spacetime/", "^kmeans_out.kea")
    
    del(results)
    del(datasets)
    del(df_res)
    del(output)
    
    # Load actual data
    
    ds.attrs['pyproj_srs'] = 'epsg:3106'
        
    # Keep track of which coordinates have been processed
    processed = []
    
    print("Generating change output for clusters...")
    
    for c in range(len(clusters)):
        
        print(c)
        
        subset = ds.salem.roi(geometry=clusters[c], crs='epsg:3106')
        
        subset = subset.dropna('x', how='all').dropna('y', how='all')
        
        curr_df = subset.to_dataframe()
        
        del(subset)
        
        curr_df = curr_df.dropna(axis=0, how='any')
        
        sub_coords = curr_df.reset_index()[['x', 'y']].drop_duplicates().to_numpy()
        
        avg_vals = curr_df.median(level='time')
        
        del(curr_df)
        
        avg_vals = avg_vals.reset_index()
        
        avg_vals['time'] = datesToNumbers(avg_vals['time'])
        
        ts_data = avg_vals.to_numpy()
        
        rows = runCOLD(ts_data, bands, output_file, False, args.re_init, ch_thresh, args.alpha)
        
        processed.extend(sub_coords.tolist())
        
        output_coords = [tuple(row) for row in sub_coords]
    
        with Pool(processes=args.processes) as pool:
            pool.starmap(writeOutPixel, output_coords)
        
    # Generate list of remaining pixels
    
    print("Generating list of remaining pixels...")
            
    all_coords = pd.DataFrame(all_coords, columns=['x', 'y'])
    
    processed = pd.DataFrame(processed, columns=['x', 'y'])
    
    all_coords = all_coords.append(processed) 
    
    all_coords = all_coords.drop_duplicates(keep=False)
    
    final_coords = [tuple(row) for row in all_coords.to_numpy()]

else:
    final_coords = all_coords
    
# Run processes for this key                                        
with Pool(processes=args.processes) as pool:
    pool.starmap(runChangeDetection, final_coords)  

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    










