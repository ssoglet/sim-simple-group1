import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import pymap3d as pm

def load_osm_ways_as_polylines(
    osm_path: str,
    min_points: int = 2,
    output_xy_order: str = "lonlat",  # "lonlat" or "latlon"
):
    ref_lat, ref_lon = 37.557557, 127.040970
    """
    Build polylines from OSM <way> using referenced <node> coordinates.

    Returns:
        List[np.ndarray], each shape (N, 2)
    """

    # ----------------------------------
    # 1. Read all nodes
    # ----------------------------------
    nodes_dict = {}  # node_id -> (lon, lat)

    for event, elem in ET.iterparse(osm_path, events=("end",)):
        if elem.tag == "node":
            node_id = elem.get("id")
            lat = elem.get("lat")
            lon = elem.get("lon")

            if node_id and lat and lon:
                nodes_dict[node_id] = (float(lon), float(lat))

        elem.clear()

    # ----------------------------------
    # 2. Read ways and build polylines
    # ----------------------------------
    polylines = []

    for event, elem in ET.iterparse(osm_path, events=("end",)):
        if elem.tag != "way":
            continue
        
        is_building = False
        for tg in elem.findall("tag"):
            tg_name = tg.get("k", None)
            if "building" in tg_name:
                is_building=True
        
        if is_building: continue
        
        coords = []
        for nd in elem.findall("nd"):
            ref = nd.get("ref")
            if ref in nodes_dict:
                coords.append(nodes_dict[ref])

        if len(coords) >= min_points:
            arr = np.asarray(coords, dtype=np.float64)
            if output_xy_order == "latlon":
                arr = arr[:, [1, 0]]
                
            # to Enu                
            east, north, _ = pm.geodetic2enu(arr[:,1], arr[:,0], 0.0, ref_lat, ref_lon, 0.0)    
            
            enu_line_ = np.array([east, north]).T
            enu_line_3d = np.hstack([enu_line_, np.zeros((enu_line_.shape[0],1))])
            polylines.append(enu_line_3d)

        elem.clear()

    return polylines

def make_osm_world_pts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    osm_path = os.path.join(base_dir, "map.osm")

    polylines = load_osm_ways_as_polylines(osm_path)
    return {
        "lane_center": polylines,
        "lane_left": None,
        "lane_right": None,
        "car_box": None
    }       

if __name__ == "__main__":
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    osm_path = os.path.join(base_dir, "map.osm")

    polylines = load_osm_ways_as_polylines(osm_path)

    print("num polylines:", len(polylines))
    print(polylines[0])

    plt.figure()
    for line_ in polylines:
        plt.plot(line_[:,0], line_[:,1])
        plt.axis('equal')
    plt.show()
