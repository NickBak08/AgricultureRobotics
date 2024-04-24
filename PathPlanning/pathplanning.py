import json
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.geometry import LineString,MultiLineString
import geopandas as gpd
import numpy as np
import pandas as pd
import math
from dubins_curves import *


def load_data(filepath,include_obstacles = False):
    """
    function to load json file and create polygon object

    args:
        filepath (str): filepath that contains the json file to be parsed
        include_obstacles (bool): boolean to indicate if you want to include the obstacles in the resulting field (will probably be usefull for some debugging)

    returns:
        field (geopandas Geoseries)

    """
    # This cell opens the json file that contains field geometry
    with open(filepath) as json_file:
        json_data = json.load(json_file) # or geojson.load(json_file)

    # This cell parses the json data and obtains a list with the coordinates for each polygon like
    # [[polygon1_coords],[polygon2_coords],....]
    # where polygon1_coords looks something like this: [[x1,y1],[x2,y2],......]
    coordinates = []
    polygons = []
    for i in range(len(json_data['features'])):
        coordinates.append(json_data['features'][i]['geometry']['coordinates'])
        polygons.append(gpd.GeoSeries(Polygon(coordinates[i][0])))
    
    field = polygons[0]
    if include_obstacles:
        for i in range(1,len(polygons)):
            field = field.symmetric_difference(polygons[i])

    return field


def generate_headlands(field,size):
    field_with_headlands = field.buffer(-size)
    return field_with_headlands
        

def linefromvec(vec):
    """
    turn a vector in to a line in 2d with a slope and intercept

    attrs:
        vec (array): 2x2 array with x,y coords of begin point in first row, xy coords of end point in second row

    returns:
        slope (float): slope of the resulting line
        intercept (float): intercept of the resulting line

    """
    slope = (vec[1][1]-vec[0][1])/(vec[1][0]-vec[0][0])
    intercept = vec[1][1]-vec[1][0]*slope
    return slope,intercept

def edge_to_line(coordinates):
    """
    turns a df of field coordinates into a df with the slopes and intercepts of all edges of the field
    
    attrs:
        coordinates (geopandas DataFrame): a GeoPandas Dataframe containing the xy coordinates of all points that describe a polygon. Can be obtained using the get_coordinates() method
    returns:
        edge_lines (pandas DataFrame): a Pandas DF that contains slopes and intercepts of all lines describing the outline of a polygon
    
    """
    edge_points = len(coordinates)
    slopes = []
    intercepts = []
    for edge in range(edge_points-1):
        x_begin = coordinates.iloc[edge]['x']
        x_end = coordinates.iloc[edge+1]['x']
        y_begin = coordinates.iloc[edge]['y']
        y_end = coordinates.iloc[edge+1]['y']
        vector = np.array([[x_begin,y_begin],[x_end,y_end]])
        slope,intercept = linefromvec(vector)
        slopes.append(slope)
        intercepts.append(intercept)
    d = {'slope': slopes, 'intercept': intercepts}
    edge_lines = pd.DataFrame(data = d)
    return edge_lines
             
def basis_AB_line(edge,coordinates):  ## TODO: make the initial position align with the actual chosen edge, right now its just randomly placed somewhere
    """
    Creates an AB line to be used as basis for filling the field, the AB line has the same orientation as the given edge  
        and the length of the AB line is such that it covers the entire y-range of the field

    attrs:
        edge (Pandas Series): a slice of a dataframe containing the edge information, with the slope and intercept information of the 
                                edge for which you want to create a base AB line
        coordinates (pandas DataFrame): a pandas DF containing all the coordinates that describe the polygon

    returns:   
        base_AB (numpy array): a line object that covers the entire y-range of the polygon with a direction
                                        specified in the edge attribute
    """
    slope = edge['slope']
    intercept = edge['intercept']
    ymin = min(coordinates['y'])
    ymax = max(coordinates['y'])
    xmin = ymin/slope-intercept
    xmax = ymax/slope-intercept
    if slope >= 0:
        vector = np.array([[xmin-xmin ,ymin],[xmax-xmin,ymax]])
    else:
        vector = np.array([[xmin-xmax ,ymin],[xmax-xmax,ymax]])
    # line = gpd.GeoSeries(LineString(vector))
    base_AB = vector
    return base_AB,slope

def fill_field_AB(base_AB,slope,coordinates,d):
    """
    takes the base AB line, its slope and a distance between lines and fills the field with AB lines of a given angle
    attrs:
        base_AB (array): array with xy coordinates of beginning and end points of the base vector
        slope (float): the slope of the basis AB line
        coordinates (pandas DataFrame): a pandas DF containing all the coordinates that describe the polygon
        d (float): distance between two AB lines

    returns:
        swath_list (list): a list containing Geoseries objects of different AB-lines
    """
    # Calculate angle of AB line wrt x-axis
    theta = np.arctan(slope)
    # using the angle and parameter d to calculate x-offset between swaths
    dx = d/(np.sin(theta)+0.01)

    # Determine amount of AB-lines that are needed
    xmax = max(coordinates['x'])
    xmin = min(coordinates['x'])
    nr_passes = int((xmax-xmin)//d + 2)*10

    # Initialize empty lists that will contain vectors and geoseries objects
    vector_list = []
    swath_list = []
    for swath in range(nr_passes):
        vector_list.append([[base_AB[0][0]+dx*swath,base_AB[0][1]],[base_AB[1][0]+dx*swath,base_AB[1][1]]])
        swath_list.append(gpd.GeoSeries(LineString(vector_list[swath])))
        # pass

    return swath_list

def clip_swaths(swath_list,field):
    """
    function to clip swaths to the headlands
    attrs:
        swath_list (list): List of geoseries objects that describe the generated swaths
        field (geopandas Geoseries): Geoseries object that describes the field geometry

    returns:
        swaths_clipped_nonempty (list): a list of geoseries objects that contains the clipped swaths that are not empty
    """
    swaths_clipped = []
    for swath in range(len(swath_list)):
        swaths_clipped.append(gpd.clip(swath_list[swath],field))
        swaths_clipped_nonempty = [swath for swath in swaths_clipped if not swath.get_coordinates().empty]
    if len(swaths_clipped_nonempty) == 0:
        raise Exception('No swaths were generated, something went wrong with clipping')
    return swaths_clipped_nonempty


def generate_path(swaths_clipped_nonempty,turning_rad,offset):
    turning_rad = 10
    line = []

    for i in range(len(swaths_clipped_nonempty)-1):
        line1 = swaths_clipped_nonempty[i]
        line2 = swaths_clipped_nonempty[i+1]
        line1_coords = line1.get_coordinates()
        line2_coords = line2.get_coordinates()

        if i%2 != 0:
            point1 = line1_coords.iloc[0]
            point2 = line2_coords.iloc[0]
            diff1 = line1_coords.iloc[0]-line1_coords.iloc[1]
            slope1 = diff1['y']/diff1['x']
            heading_1 = math.degrees(math.atan(slope1))+180

            diff2 = line2_coords.iloc[0]-line2_coords.iloc[1]
            slope2 = diff2['y']/diff1['x']
            heading_2 = math.degrees(math.atan(slope2))
            pt1 = (line1_coords.iloc[0]['x'],line1_coords.iloc[0]['y'],90-heading_1+offset)
            pt2 = (line2_coords.iloc[0]['x'],line2_coords.iloc[0]['y'],90-heading_2+offset)
        else:
            point1 = line1_coords.iloc[1]
            point2 = line2_coords.iloc[1]
            diff1 = line1_coords.iloc[0]-line1_coords.iloc[1]
            slope1 = diff1['y']/diff1['x']
            heading_1 = math.degrees(math.atan(slope1))

            diff2 = line2_coords.iloc[0]-line2_coords.iloc[1]
            slope2 = diff2['y']/diff1['x']
            heading_2 = math.degrees(math.atan(slope2))+180
            pt1 = (line1_coords.iloc[1]['x'],line1_coords.iloc[1]['y'],90-heading_1+offset)
            pt2 = (line2_coords.iloc[1]['x'],line2_coords.iloc[1]['y'],90-heading_2+offset)



        path = dubins_main(pt1,pt2,turning_rad)
        curve1 = LineString(path[:,0:2])
        line.append(gpd.GeoSeries((line1[0])))
        line.append(gpd.GeoSeries(curve1))
    line.append(gpd.GeoSeries(line2[0]))
    return line
        
def path_to_df(best_path):
    """
    generates a df with the coordinates from the best path

    args:
        best_path (list): a list of linestring objects that define a path

    returns: 
        df_best_path (pandas DataFrame): a dateframe with x and y coordinates of the best path
    """
    coords = [x.get_coordinates() for x in best_path]
    df = pd.concat(coords)
    df_best_path = df.set_index((np.arange(len(df))))
    return df_best_path


def interpolate_path(path,distance):
    """
    Function to interpolate the straight line segments of the path (otherwise they are simply a beginning and endpoint)
    args:
        path (list): list of geopandas Geoseries objects that describe the path
        distance (float): distance between interpolated points

    returns:
        path (list): list of geopandas Geoseries objects that has the same length as the input, 
                        but each straight line segment is split into more pieces
    """


    for i in range(len(path)): # Loop over the list of path segments
        if i%2 == 0: # We always start the path with a straight line, and then turns and straight lines alternate, so each even entry in the list is a straight line segment
            if i%4 ==0: # Every second straight line segment has to be reversed; each vector has the same direction, but the tractor
                        #should drive back and forth, so each 4th element in the row (every second straight path) reversed and then interpolated
                line = path[i][0]
                distances = np.arange(0,line.length,distance)
                interpolated_path = LineString([line.interpolate(distance) for distance in distances])
                path[i] = gpd.GeoSeries(interpolated_path)
            else:
                line = path[i][0]
                distances = np.arange(line.length,0,-distance)
                interpolated_path = LineString([line.interpolate(distance) for distance in distances])
                path[i] = gpd.GeoSeries(interpolated_path)
        else: # Don't do anything if we have an odd index, those correspond to turns which are already interpolated. 
            continue
    return path


def pathplanning(data_path,include_obs,turning_rad,distance,plotting,headland_size, interpolation_dist):
    """
    main function for the pathplanning, loads the data and generates a path (maybe its better to make this a class but idk)
    
    args:
        data_path (str): filepath that contains the json file to be parsed
        include_obs (bool): boolean to indicate if you want to include the obstacles in the 
                            resulting field (will probably be usefull for some debugging)
        turning_rad (float): turning radius of the tractor in m
        distance (float): distance between swaths in m
        plotting (bool): boolean to decide whether you want to plot the generated path or not
        headland_size (float): size of the headlands in m
        interpolation_dist (float): the distance between points in the straight line segment

    returns:
        field (geopandas Geoseries): The polygon that defines the field
        best_path (pandas DataFrame): Dataframe containing the xy coordinates of each checkpoint of the path. 

    """
    field = load_data(data_path,include_obs) 
    field_headlands = generate_headlands(field,headland_size)
    coordinates = field_headlands.get_coordinates()
    lines = edge_to_line(coordinates)
    
    # Initialize emtpy lists to contain the generated paths and path lengths
    paths = [] 
    path_lengths = []

    for i in range(len(coordinates)-1): # Loop over all of the edges
        print('Finding path {}/{}'.format(i,len(coordinates)-1))
        try:
            # Calculate the basis AB line, use that to fill the field and clip the swaths
            line,slope = basis_AB_line(lines.iloc[i],coordinates)
            swath_list = fill_field_AB(line,slope,coordinates,distance)
            swaths_clipped = clip_swaths(swath_list,field_headlands)
            
            # If the paths have a negative slope, the heading is off by 180 degrees, this messes up the curves, so this offset is introduced
            if slope > 0:
                offset = 0
            else:
                offset = 180
            
            # Make a path with dubins curves
            path = generate_path(swaths_clipped,turning_rad,offset)
            # Interpolate the path with some specified distance
            path = interpolate_path(path,interpolation_dist)
            paths.append(path)

            # Calculate path lenght as a measure of how good a path is, this should be replaced with a different measure
            total_len = 0
            for i in range(len(path)):
                total_len+= path[i].length.item()
            path_lengths.append(total_len)
        except Exception as error: # Error handling to make sure the loop continues if an error pops up
            paths.append(None)
            path_lengths.append(0)
            print(error)
            continue

    # Finding the path with the best measure
    best_path_index = np.argmax(path_lengths)
    best_path = paths[best_path_index]
    # Converting path to df
    best_path = path_to_df(best_path)

    # Plot the path if it is specified
    if plotting:
        fig, ax = plt.subplots()
        field.plot(ax = ax,color = 'g')
        field_headlands.boundary.plot(ax = ax,color = 'r')
        best_path.plot(x = 'x', y = 'y',ax = ax,color = 'magenta')
        plt.show()
    
    return field,best_path





data_path ="./data/field_geometry/test_2.json"
include_obs = False
turning_rad = 10
distance = 20
field, best_path = pathplanning(data_path,include_obs,turning_rad,distance,True,25,5)



