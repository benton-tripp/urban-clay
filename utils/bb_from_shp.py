import geopandas as gpd
from shapely.geometry import box

def get_bbox_from_shp(shapefile_path, return_wkt=False, return_min_max=False):
    # Read the shapefile
    gdf = gpd.read_file(shapefile_path)
    
    # Reproject to WGS84 if not already in that CRS
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)
        # print("Reprojected to WGS84 (EPSG:4326)")
    
    # Get the bounding box coordinates (minx, miny, maxx, maxy)
    bbox = gdf.total_bounds  
    
    # Check for None values in bbox
    if any(coord is None for coord in bbox):
        raise ValueError("Bounding box contains None values. Check your shapefile.")
    
    if return_wkt:
        polygon = box(bbox[0], bbox[1], bbox[2], bbox[3])  # minx, miny, maxx, maxy
        # Convert Polygon to WKT format
        bbox = polygon.wkt
    elif return_min_max:
        bbox = gdf.total_bounds  # (minx, miny, maxx, maxy)
    else:
        # Format bounding box for use with features_from_bbox (north, south, east, west)
        bbox = (bbox[3], bbox[1], bbox[2], bbox[0])  # (north, south, east, west)
    # print("BBox:")
    # print(bbox)
    return bbox