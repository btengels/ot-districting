import os
import geopandas as geo
import shapely
import numpy as np


def make_folder(path):
    """
    This function makes directories on the fly at location 'path'. Checks first 
    to see if the directory is already there. If not, it makes the directory.

    INPUTS
    ----------------------------------------------------------------------------    
    path: string, path where directory will be made

    OUTPUTS
    ----------------------------------------------------------------------------    
    None
    """
    try: 
        # check to see if path already exists otherwise make folder
        os.makedirs(path)

    except OSError:
        if not os.path.isdir(path):
            raise

    return None


def unpack_multipolygons(geo_df, impute_vals=True):
    """
    Takes a vector of polygons and/or multipolygons. Returns an array of only 
    polygons by taking multipolygons and putting the underlying polygons in a 
    new vector. The "districts" vector contains info on the district for each 
    multipolygon/polygon. 

    INPUTS: 
    ----------------------------------------------------------------------------
    geo: numpy array, array of shapely polygons/multipolygons
    impute_vals: boolean(default=True) use area shares to impute populations or
                 other variabels for polygons within a multipolygon. If false,
                 population/political variables are duplicated across all 
                 polygons within a multipolygon. 
    
    OUTPUTS:
    ----------------------------------------------------------------------------
    geo: numpy array, array of shapely polygons only, (no multipolygons)
    dists: numpy array, vector contianing info on each polygon
    """
    repeat = False
    new_df = geo.GeoDataFrame()
    new_df.crs = geo_df.crs

    cols = [c for c in geo_df.columns if 'POP' in c or 'VAP' in c or 'PRE' in c]
    geo_df[cols] = geo_df[cols].astype(float)
    scale = 1

    # check geometry column for non-polygon objects (multipolygons)
    for i, row in geo_df.iterrows():
        row = row.T
        if type(row.geometry) == shapely.geometry.polygon.Polygon:
            new_df = new_df.append(row)

        else:
            # unpack multipolygons, perhaps impute numerical variables
            multi_row = row.copy()
            total_area = multi_row.geometry.area            
            for poly in multi_row.geometry:

                if impute_vals is True:
                    scale = (poly.area/total_area)

                row.geometry = poly
                row[cols] *= scale
                row.ALAND10 *= scale
                row.area = poly.area        
                new_df = new_df.append(row)

    poly_true = np.array([g == 'Polygon' for g in new_df.geom_type])
    if poly_true.any() is False:
        new_df = unpack_multipolygons(new_df)

    return new_df   


def get_coords(T, xcoord=True):
    """
    Takes a single shapely polygon and returns a list of its x or y coordinates.

    INPUTS
    ----------------------------------------------------------------------------
    T: shapely Polygon or MultiPolygon, voter precinct or district
    xcoord: boolean (default=True), if true returns x coordinates, if false 
            returns y coordinates

    OUTPUT
    ----------------------------------------------------------------------------    
    patchx: list, sequence of x coordinates corresponding to patchy
    patchy: list, sequence of y coordinates corresponding to patchx
    """
    # if type(T) == shapely.geometry.polygon.Polygon:
    

    if type(T) == shapely.geometry.multipolygon.MultiPolygon:
        T = T[0]
    # patchx, patchy = T.exterior.coords.xy
    patchx, patchy = T.exterior.coords.xy
    
    # can choose to return either the x or y coordinates 
    # (this odd but is allows for pandas' "apply" function)
    if xcoord is True:
        return list(patchx)
    else:
        return list(patchy)
