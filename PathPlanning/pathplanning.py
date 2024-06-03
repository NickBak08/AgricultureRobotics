import json
import matplotlib.pyplot as plt
import matplotlib
from shapely.geometry import Polygon
from shapely.geometry import LineString,MultiLineString
import geopandas as gpd
import numpy as np
import pandas as pd
import math
from .dubins_curves import *
import shapely.affinity
from shapely import Point
from shapely.ops import split, nearest_points


def load_data(filepath,include_obstacles = False,scale_pixels:int=1):
    """
    Loads json file and create polygon object for the field.

    Parameters:
        filepath (str): filepath that contains the json file to be parsed
        include_obstacles (bool): boolean to indicate if you want to include the obstacles in the resulting field (will probably be usefull for some debugging)
        scale_pixels (int): to scale the field to the real size

    Returns:
        field (geopandas GeoSeries): GeoSeries of the polygon - the shape of the field (with obstacles if include_obstacles = True)
    """

    with open(filepath) as json_file:
        json_data = json.load(json_file)

    coordinates = []
    polygons = []
    for i in range(len(json_data['features'])):
        coordinates.append(json_data['features'][i]['geometry']['coordinates'])
        polygons.append(gpd.GeoSeries(Polygon(coordinates[i][0])).scale(scale_pixels,scale_pixels))
    
    field = polygons[0]
    obstacles = []

    if include_obstacles:
        for i in range(1,len(polygons)):
            obstacles.append(polygons[i])
        return field,obstacles
    else:
        return field


def generate_headlands(field,turning_rad,tractor_width):
    """
    Generates headlands around the field and obstacles.

    Parameters:
        field (geopandas GeoSeries): GeoSeries of the polygon - the shape of the field (possibly with obstacles)
        turning_rad (float): turning radius of the tractor in m
        tractor_width (float): width of the tractor in m

    Returns:
        field_with_headlands (geopandas GeoSeries): GeoSeries of the polygon, made smaller with the buffer 
        headland_size (float): width of the buffer in m
    """
    headland_size = 3*turning_rad # minimum width of the headlands for the tractor to turn
    headland_size = math.ceil(headland_size / tractor_width) * tractor_width # make sure the headlands are a multiple of the tractor width to not waste space or drive over seeds
    field_with_headlands = field.buffer(-headland_size, cap_style = 'square', join_style = 'mitre')
    
    return field_with_headlands, headland_size
        

def linefromvec(vec):
    """
    Turns a vector in to a line in 2d with a slope and intercept.

    Parameters:
        vec (array): 2x2 array with x,y coordinates of begin point in first row, xy coordinates of end point in second row

    Returns:
        slope (float): slope of the resulting line
        intercept (float): intercept of the resulting line

    """
    if vec[1][0]-vec[0][0] == 0: # if the line is vertical, the y-intercept and slope don't exist
        slope = None
        intercept = vec[0][0] # store the x-intercept instead of the y-intercpet
    else:
        slope = (vec[1][1]-vec[0][1])/(vec[1][0]-vec[0][0]) # if the line is horizontal, the slope is 0
        intercept = vec[1][1]-vec[1][0]*slope 

    return slope,intercept

def edge_to_line(coordinates):
    """
    Turns a DataFrame of field coordinates into a DataFrame with the slopes and intercepts of all edges of the field.
    
    Parameters:
        coordinates (geopandas DataFrame): a GeoPandas Dataframe containing the xy coordinates of all points that describe a polygon. Can be obtained using the get_coordinates() method
    Returns:
        edge_lines (pandas DataFrame): a Pandas DataFrame that contains slopes and intercepts of all lines describing the outline of a polygon
    
    """
    edge_lines = pd.DataFrame(columns=['slope', 'intercept'])

    for edge in range(len(coordinates)-1):
        x_begin = coordinates.iloc[edge]['x']
        x_end = coordinates.iloc[edge+1]['x']
        y_begin = coordinates.iloc[edge]['y']
        y_end = coordinates.iloc[edge+1]['y']
        vector = np.array([[x_begin,y_begin],[x_end,y_end]])
        slope,intercept = linefromvec(vector)
        edge_lines = pd.concat([edge_lines, pd.DataFrame([[slope, intercept]], columns=['slope', 'intercept'])], ignore_index=True)

    return edge_lines

def basis_AB_line(edge,coordinates):  
    """
    Creates an AB line to be used as basis for filling the field, the AB line has the same orientation as the given edge  
        and the length of the AB line is such that it covers the entire y-range of the field.

    Parameters:
        edge (Pandas Series): a slice of a DataFrame containing the edge information, with the slope and intercept information of the 
                                edge for which you want to create a base AB line
        coordinates (pandas DataFrame): a pandas DataFrame containing all the coordinates that describe the polygon

    Returns:   
        base_AB (numpy array): a line object that covers the entire y-range of the polygon with a direction specified in the edge attribute
        slope (float): slope of the base AB-line 
    """
    slope = edge['slope']
    intercept = edge['intercept']
    xmin = min(coordinates['x'])
    xmax = max(coordinates['x'])
    ymin = min(coordinates['y'])
    ymax = max(coordinates['y'])

    if pd.isna(slope): # for vertical lines 
        base_AB = np.array([[intercept, ymin],[intercept, ymax]]) # edge x-coord stay the same, y stretches to ymin and ymax
    else: 
        if slope > 0:
            x_1 = min(xmin, (ymin-intercept)/slope)
            x_2 = max(xmax, (ymax-intercept)/slope)
        elif slope == 0:
            x_1 = xmin
            x_2 = xmax  
        elif slope < 0:
            x_1 = min(xmin, (ymax-intercept)/slope)
            x_2 = max(xmax, (ymin-intercept)/slope)
        y_1 = x_1*slope+intercept   
        y_2 = x_2*slope+intercept
        base_AB = np.array([[x_1, y_1],[x_2, y_2]])

    return base_AB, slope

def fill_field_AB(base_AB,slope,coordinates,tractor_width):
    """
    Takes the base AB-line, its slope and a distance between lines (equal to tractor width) and fills the field with AB-lines of a given angle.
    
    Parameters:
        base_AB (array): array with xy coordinates of beginning and end points of the base vector
        slope (float): the slope of the basis AB-line
        coordinates (pandas DataFrame): a pandas DataFrame containing all the coordinates that describe the polygon
        tractor_width (float): tractor width (distance between two AB-lines) in m

    Returns:
        swath_list (list): a list containing Geoseries objects of different AB-lines
    """
    
    # Initialize empty lists that will contain geoseries objects
    swath_list = []

    # Determine amount of AB-lines that are needed if they are not horizontal
    xmax = max(coordinates['x'])
    xmin = min(coordinates['x'])
    nr_passes = int((xmax-xmin)//tractor_width + 2)

    if pd.isna(slope): # vertical AB-lines
        dx = tractor_width
        base_AB[0][0]+= dx/2 # update the base AB line to be inside of the mainland and the planter can plant exactly on the headland line
        base_AB[1][0]+= dx/2
        for i in range(-nr_passes,nr_passes):
            swath_list.append(gpd.GeoSeries(LineString([[base_AB[0][0]+dx*i,base_AB[0][1]],[base_AB[1][0]+dx*i,base_AB[1][1]]])))
    else:
        theta = np.arctan(slope) # angle of AB line wrt x-axis
        if theta == 0 : # horizontal AB line
            dy = tractor_width # use y-offset for horizontal lines
            base_AB[0][1]+= dy/2 # update the base AB line to be inside of the mainland and the planter can plant exactly on the headland line
            base_AB[1][1]+= dy/2
            # Determine amount of AB-lines that are needed for horizontal lines
            ymax = max(coordinates['y'])
            ymin = min(coordinates['y'])
            nr_passes = int((ymax-ymin)//tractor_width + 2)
            # update the list of AB-lines 
            for i in range(-nr_passes,nr_passes):
                swath_list.append(gpd.GeoSeries(LineString([[base_AB[0][0],base_AB[0][1]+dy*i],[base_AB[1][0],base_AB[1][1]+dy*i]])))
        else:
            dx = tractor_width/np.sin(theta) # use the angle and parameter d to calculate x-offset between swaths
            base_AB[0][0]+= dx/2 # update the base AB line to be inside of the mainland and the planter can plant exactly on the headland line
            base_AB[1][0]+= dx/2
            for i in range(-nr_passes,nr_passes):
                swath_list.append(gpd.GeoSeries(LineString([[base_AB[0][0]+dx*i,base_AB[0][1]],[base_AB[1][0]+dx*i,base_AB[1][1]]])))    

    return swath_list

def clip_swaths(swath_list,field):
    """
    Clips AB-lines to the headlands.

    Parameters:
        swath_list (list): List of GeoSeries objects that describe the generated AB-lines
        field (geopandas Geoseries): GeoSeries object that describes the field geometry

    Returns:
        swaths_clipped_nonempty (list): a list of GeoSeries objects that contains the clipped AB-lines that are not empty
    """
    swaths_clipped = []

    for swath in range(len(swath_list)):
        swaths_clipped.append(gpd.clip(swath_list[swath],field))
        swaths_clipped_nonempty = [swath for swath in swaths_clipped if not swath.get_coordinates().empty]
        swaths_type = [type(swath[0]) for swath in swaths_clipped_nonempty]
    if len(swaths_clipped_nonempty) == 0:
        raise Exception('No swaths were generated, something went wrong with clipping')
    if MultiLineString in swaths_type:
        raise Exception('No swaths were generated, need to decompose the field.')
    
    return swaths_clipped_nonempty

def connect_paths(path1,path2,turning_rad):
    """
    Connects two AB-lines with a Dubins curve, returns just the curve.

    Parameters: 
        path1 (geopandas GeoSeries): GeoSeries object which is a LineString representing the first line to be connected 
        path2 (geopandas GeoSeries): GeoSeries object which is a LineString representing the second line to be connected
        turning_rad (float): turning radius of the tractor in m

    Returns:
        curve (LineString): a LineString object representing the curve between the two lines to be connected 
    """

    # get coordinates of the lines to calculate the heading and the points to be connected by the curve
    path1_coords = path1.get_coordinates()
    path2_coords = path2.get_coordinates()
    begin_point = path1_coords.iloc[-1]
    end_point = path2_coords.iloc[0]

    # caluclate the differences in coordinates for both lines to calculate the headings of the lines
    # the headings are calculated with respect to the positive part of the y-axis
    diff1 =  path1_coords.iloc[-2]-path1_coords.iloc[-1]
    if diff1['x'] == 0 and diff1['y'] > 0: # the line is vertical, going down
        heading1 = 180
    elif diff1['x'] == 0 and diff1['y'] < 0: # the line is vertical, going up
        heading1 = 0     
    elif diff1['y'] == 0 and diff1['x']> 0: # If the line is horizontal, going left
        heading1 = 270
    elif diff1['x'] >0 : # if the heading is right (up or down), because atan is between -90 and 90
        heading1 = 270-math.degrees(math.atan(diff1['y']/diff1['x']))
    else: # if the heading is left (up or down), because atan is between -90 and 90
        heading1 = 90-math.degrees(math.atan(diff1['y']/diff1['x']))
    
    diff2 = path2_coords.iloc[0]-path2_coords.iloc[1]
    if diff2['x'] == 0 and diff2['y']>0: # the line is vertical
        heading2 = 180
    elif diff2['x'] == 0 and diff2['y'] < 0: # the line is vertical
        heading2 = 0
    elif diff2['y'] == 0 and diff2['x']> 0: # If the line is horizontal, going left
        heading2 = 270
    elif diff2['x'] >0: # if the heading is right (up or down), because atan is between -90 and 90
        heading2 = 270-math.degrees(math.atan(diff2['y']/diff2['x']))
    else: # if the heading is left (up or down), because atan is between -90 and 90
        heading2 = 90-math.degrees(math.atan(diff2['y']/diff2['x']))

    # make the curve connecting the two points
    pt1 = (begin_point.x,begin_point.y,heading1)
    pt2 = (end_point.x,end_point.y,heading2)
    path = dubins_main(pt1,pt2,turning_rad)
    curve = LineString(path[:,0:2])

    return curve

def generate_path(swaths_clipped_nonempty, turning_rad):
    """
    Generate path for the tractor in the mainland by connecting clipped AB-lines.

    Parameters:
        swaths_clipped_nonempty (list): a list of GeoSeries objects that contains the clipped AB-lines 
        turning_rad (float): turning radius of the tractor in m

    Returns:
        path (list): a list of GeoSeries objects that contains pieces of the path in the mainland
    """

    # initialize the empty list of parts of the final path
    path = []

    for i in range(len(swaths_clipped_nonempty)-1):
        if i%2 == 0:
            # obtain two consecutive lines from the list of clipped AB-lines
            line1 = swaths_clipped_nonempty[i]
            line2 = shapely.reverse(swaths_clipped_nonempty[i+1])
            # make a curve between two lines 
            curve = connect_paths(line1, line2, turning_rad)
            # append a line and the curve connecting it to the next line to the final path 
        else:
            # obtain two consecutive lines from the list of clipped AB-lines
            line1 = shapely.reverse(swaths_clipped_nonempty[i])
            line2 = swaths_clipped_nonempty[i+1]
            # make a curve between two lines 
            curve = connect_paths(line1, line2, turning_rad)
            # append a line and the curve connecting it to the next line to the final path 
        path.append(gpd.GeoSeries((line1[0])))
        path.append(gpd.GeoSeries(curve))

    # append the last AB-line to the path 
    path.append(gpd.GeoSeries(line2[0]))

    return path

def interpolate_path(path,distance,base,no_offset):
    """
    Interpolates the straight line segments of the path and indicates in which point the seeds should be planted.
    
    Parameters:
        path (list): list of geopandas Geoseries objects that describe the path
        distance (float): distance between interpolated points
        base (LineString): reference line for planting seeds 
        no_offset (bool): boolean to indicate whether the seeds need to be aligned, if False, seeds will be aligned

    Returns:
        path (list): list of geopandas GeoSeries objects that has the same length as the input, 
                        but each straight line segment is split into more pieces
        planting (list): list of 1s and 0s indicating if we plant seeds in a given point or not
    """
    planting = []
    
    for i in range(len(path)): # Loop over the list of path segments
        if i%2 == 0: # We always start the path with a straight line, and then turns and straight lines alternate, so each even entry in the list is a straight line segment
            line = path[i][0]
            if i%4 ==0: # Every second straight line segment has to be reversed; each vector has the same direction, but the tractor
                        #should drive back and forth, so each 4th element in the row (every second straight path) reversed and then interpolated
                begin = Point(line.coords[0]) # beginning of the straight line on tractor's path
                distance_from_base  = begin.distance(base) # distance from the reference line
                offset = distance - distance_from_base%distance # how much the seeding needs to be offset to make the seeds aligned
                if no_offset:
                    offset = 0
                distances = np.arange(offset,line.length,distance) # distances between the placed seeds and the beginning of the seed grid
            else:
                begin = Point(line.coords[0]) # beginning of the straight line on tractor's path - the other direction than the previous one
                distance_from_base  = begin.distance(base)
                offset = distance_from_base%distance # the other way than the previous line
                if no_offset:
                    offset = 0
                distances = np.arange(offset,line.length,distance) # the other way than the previous line

            try: # if the line is long enough to interpolate
                interpolated_path = LineString([line.interpolate(distance) for distance in distances])
            except: # if the line is not long enough to interpolate, keep the line as it is
                interpolated_path = line

            length = len(interpolated_path.coords)
            path[i] = gpd.GeoSeries(interpolated_path)
            planting.append([1]*length) # 1 means we plant in this spot

        else: # Don't do anything if we have an odd index, those correspond to turns which are already interpolated. 
            planting.append(len(path[i][0].coords)*[0]) # 0 means we don't plant in this spot
            continue

    planting = [val for sublist in planting for val in sublist] # flatten the list of 0s and 1s indicating if we plant or not

    return path,planting
        
def path_to_df(best_path,planting):
    """
    Generates a DataFrame with the coordinates from the best path.

    Parameters:
        best_path (list): a list of LineString objects that define a path
        planting (list): a list of 1s and 0s indicating if we plant seeds in a given point or not

    Returns: 
        df_best_path (pandas DataFrame): a DataFrame with x and y coordinates of the best path and a command to plant in a given point or not
    """

    df = pd.concat([x.get_coordinates() for x in best_path])
    df['command'] = planting
    df_best_path = df.set_index((np.arange(len(df))))

    return df_best_path

def find_ref(swaths_clipped, field_headlands,slope):
    """
    Finds a reference line for the seed positions. The grid will be constant with respect to this line.

    Parameters: 
        swaths_clipped (list geopandas.GeoSeries): list of the clipped AB-lines that cover the field
        field_headlands (geopandas.GeoSeries): a geoseries objec that defines the field after substracting the healands
        slope (float): a float that describes the slope of the AB-lines

    Returns:
        scaled_normal (LineString): reference line scaled to cover the whole field 
    """ 

    # If the slope is > 1, we have a 'steep' swath and we need it's minimum y, otherwise use min x
    if np.abs(slope)>=1 or np.isnan(slope):
        # Make a list of starting coordinates for all swaths, we then take the minimum of those values generate a line that is normal to the swaths
        start_y = [item[0].coords[0][1] for item in swaths_clipped]
        index = np.argmin(np.array(start_y))
    else:
        # Make a list of starting coordinates for all swaths, we then take the minimum of those values generate a line that is normal to the swaths
        start_x = [item[0].coords[0][0] for item in swaths_clipped]
        index = np.argmin(np.array(start_x))
        
    reference_swath = swaths_clipped[index] # get the reference swath and rotate it by 90 degrees around its starting point
    reference_normal = shapely.affinity.rotate(reference_swath[0],90,origin= Point(reference_swath[0].coords[1]))

    # Calculate the width and heigth of the field to find the factor with which we have to scale the reference line to make sure it covers the entire field
    field_width = np.max(field_headlands.boundary[0].xy[0])-np.min(field_headlands.boundary[0].xy[0])
    field_heigth = np.max(field_headlands.boundary[0].xy[1])-np.min(field_headlands.boundary[0].xy[1])
    scale_factor = np.max([field_width,field_heigth])

    # If we have a steep slope, we move the reference downwards to make sure it is outside the field
    # Otherwise we move it to the left for the same reason
    if np.abs(slope)>=1 or np.isnan(slope):
        reference_normal = shapely.affinity.translate(reference_normal,0,-field_heigth)
    else:
        reference_normal =  shapely.affinity.translate(reference_normal,-field_width,0)

    # Scale the ref line with the factor defined above
    scaled_normal = shapely.affinity.scale(reference_normal,xfact = scale_factor,yfact = scale_factor)

    return scaled_normal

def plantseeds(best_path,tractor_width,seed_distance):
    """ 
    Finds the lines in which seeds are planted and exact seed positions.

    Parameters:
        best_path (pandas DataFrame): a DataFrame with x and y coordinates of the best path and command indicating if the seeds are planted (1) or not (0) in the given point
        tractor_width (float): the width of the tractor in m
        seed_distance: distance between each neighbouring seed within one line of seeds planted at once in m

    Returns:
        seed_lines (list): list of of LineStrings and None - the lines of seeds planted at once, or None corresponding to the 'command' for each point in tractor's path
        seed_positions (list of MultiPoints): positions of each seed within a line of seeds
    """

    # initialize the lists of seed lines and positions
    seed_lines = []
    seed_positions = []
    distances = np.arange(0.5*seed_distance,tractor_width+0.5*seed_distance,seed_distance) # distances between seeds within one tractor width
    seed_count = len(distances)
    for index,row in best_path.iterrows():
        if row['command'] == 1: # if we are supposed to plant in a given place
            if index == len(best_path)-1:
                continue
            # the line of the tractor
            swath_vector = LineString([[0,0],[best_path['x'].iloc[index+1]-best_path['x'].iloc[index],
                                              best_path['y'].iloc[index+1]-best_path['y'].iloc[index]]]) 
            # planting is perpendicular to the line of the tractor and on both sides of the tractor 
            swath_normal = LineString([[-swath_vector.coords[1][1]/swath_vector.length*tractor_width/2,
                                        swath_vector.coords[1][0]/swath_vector.length*tractor_width/2],
                                        [swath_vector.coords[1][1]/swath_vector.length*tractor_width/2,
                                         -swath_vector.coords[1][0]/swath_vector.length*tractor_width/2]]) 
            seed_lines.append(shapely.affinity.translate(swath_normal,row['x'],row['y']))
            seed_positions.append([seed_lines[index].interpolate(distance) for distance in distances])
        else:
            seed_lines.append(None)

    seed_positions = shapely.MultiPoint([val for sublist in seed_positions for val in sublist]) # flatten the list of seed positions
    
    return seed_count, seed_positions

def generate_connect_headlands(field,mainland_path,tractor_width,turning_rad,headland_size,seed_distance):
    """ 
    Generate paths in the headlands and connect them together and to the mainland path.

    args:
        field (geopandas Geoseries): Geoseries of Polygons representing the field
        mainland_path (pandas DataFrame): DataFrame of the chosen path for the mainland
        tractor_width (float): the width of the tractor in m 
        turning_rad (float): the turning radius of the tractor in m 
        seed_distance (float): distance between lines of seeds, same as between interpolated points, in m

    returns:
       connected_path_df (pandas DataFrame): DataFrame of paths of the tractor in the headlands and commands to plant seeds
    """
    
    field = field[0] # it needs to be a polygon, and field is a geoseries
    paths = []
    directions = [-1]  # direction flag to alternate direction, stored in a list for connecting the paths 
    command_list = [] # command for planting
    nr_passes_needed = headland_size//tractor_width

    for i in range(nr_passes_needed):
        path = field.buffer(-1*(tractor_width/2+i*tractor_width))
        # buffer to smooth the path
        path_rounded = path.buffer(turning_rad).buffer(-2 * turning_rad).buffer(turning_rad) 
        
        if isinstance(path_rounded, Polygon):
            exterior = path_rounded.exterior
        elif isinstance(path_rounded, list):
            exterior = path_rounded[0].exterior
        else:
            break

        # convert exterior to LineString
        if directions[-1] == 1:
            path_line = LineString(exterior.coords)
        else:
            path_line = LineString(exterior.coords[::-1]) # reverse the direction for connected path
        
        # interpolate the paths for planting seeds
        distances = np.arange(0,int(path_line.length),seed_distance)

        try: # if the line is long enough to interpolate
            interpolated = LineString([path_line.interpolate(distance) for distance in distances])
        except: # if the line is not long enough to interpolate, keep the line as it is
            interpolated = path_line

        paths.append(interpolated)
        directions.append(-1 * directions[-1])  # alternate direction for the next path
    paths = reversed(paths) # reverse the order of the path so that they go towards the outside of the field
    paths = [path for path in paths if not path.is_empty] # make sure the paths are non-empty

    # Connect the paths
    connected_path = []
    for i in range(len(paths)):
        if i == 0:
            # the first path in the headlands need to be connected to the path in the mainland
            end_point = Point(mainland_path[-1].get_coordinates().iloc[-1])
            # print(len(mainland_path))
            path1 = mainland_path[-1]
        else:
            end_point = Point(connected_path[-1].get_coordinates().iloc[-1])
            path1 = connected_path[-1]
        
        # end point of the connection, which is the first point of the next path
        _,closest_point  = nearest_points(end_point,paths[i])
        result = split(paths[i],closest_point.buffer(0.1))
        path2 = gpd.GeoSeries(result.geoms[1])

        # connect the paths
        connection = connect_paths(path1,path2,turning_rad) # this is a LineString
        connected_path.append(gpd.GeoSeries(connection))  # paths need to be GeoSeries to be used in path_to_df
        # connected_path.append(gpd.GeoSeries(paths[i]))
        connected_path.append(gpd.GeoSeries(result.geoms[1]))
        connected_path.append(gpd.GeoSeries(result.geoms[2]))
        connected_path.append(gpd.GeoSeries(result.geoms[0]))

        # update the commands for planting
        command_list.append([0]*len(connection.coords)) # no planting on the turns
        # command_list.append([1]*len(paths[i].coords)) # planting seeds on the straight lines, note that the ith path is used here
        command_list.append([1]*(len(result.geoms[1].coords)+len(result.geoms[2].coords)+len(result.geoms[0].coords)))

    command_list = [val for sublist in command_list for val in sublist]

    connected_path_df = path_to_df(connected_path,command_list)

    return connected_path_df

def create_grid(sp,seed_count,seed_distance,tractor_width):
    """
    Creates a grid to evaluate the alignment of the seeds, the grid is based on the first 2 planted rows of seeds.

    Parameters:
        sp (geopandas.GeoSeries): a GeoSeries object that contains all seed locations
        seed_count (int): number of seeds in one planted row
        seed_distance (float): distance between seeds in a row and between different rows in m
        tractor_width (float): width of the tractor in m
        
    Returns:
        grid_hor (MultiLineString): MultiLineString object containing the horizontal lines of the grid
        grid_ver (MultiLineString): MultiLineString object containing the vertical lines of the grid
    """

    # Create a dataframe of seed coordinates
    sp_df = gpd.GeoSeries(sp).get_coordinates()

    # Pick the points of 3 seeds, coming from the first 2 planted rows
    pointa = Point(sp_df.iloc[0])
    pointb = Point(sp_df.iloc[1])
    pointc = Point(sp_df.iloc[seed_count])

    # Creating horizontal and vertical line segments that define the grid
    line_seg_hor = LineString([pointa,pointc])
    line_seg_ver = LineString([pointa,pointb])

    # Calculating field width to scale the line segments
    field_width = max(sp_df['x'])-min(sp_df['x'])
    field_heigth = max(sp_df['y'])-min(sp_df['y'])
    
    # A safety margin to make sure the grid covers the entire field
    safety_marg = 10

    # Scaling a line segment based on field width and a safety margin
    factor = max([field_width,field_heigth])*safety_marg
    line_hor = shapely.affinity.scale(line_seg_hor,xfact = factor,yfact = factor)
    line_ver = shapely.affinity.scale(line_seg_ver,xfact = factor,yfact = factor)

    # Intialize empty lists for the horizontal and vertical lines
    lines_hor = []
    lines_ver = []

    # Calculate the nr of lines that are needed for covering the field (taking into account the safety margin)
    nr_lines_hor = int(factor//line_seg_hor.length*safety_marg)
    nr_lines_ver = int(factor//line_seg_ver.length*safety_marg)
    
    # Fill the field based on the required number of lines to form the grid
    for i in range(-nr_lines_hor,nr_lines_hor):
        lines_hor.append(line_hor.parallel_offset(seed_distance*i,'left'))
    for i in range(-nr_lines_ver,nr_lines_ver):
        lines_ver.append(line_ver.parallel_offset(seed_distance*i,'right'))
    
    # Create multilinestrings from the grids
    grid_hor = MultiLineString(lines_hor)
    grid_ver = MultiLineString(lines_ver)

    return grid_hor,grid_ver,seed_count


def check_alignment(sp,grid_hor,grid_ver,seed_count,margin = 0.01,full_score = False):
    """
    Checks if the seeds are aligned to the grid.

    Parameters:
        sp (geopandas GeoSeries): a geoseries object that contains all seed locations
        grid_hor (MultiLineString): MultiLineString object containing the horizontal lines of the grid
        grid_ver (MultiLineString): MultiLineString object containing the vertical lines of the grid
        seed_count (int): number of seeds in one planted row
        margin (float) : accuracy margin for when the seed is considered to be 'on the grid'
        full_score (bool): boolean to indicate whether the accuracy is calculated for all seeds or just 1 seed per row

    Returns:
        score (float): a score between 1 and 0 indicating how many seeds are on the grid
    """

    # If statement to check if we want to calculate all seeds, or just 1 per row
    if full_score:
        index_seeds_to_check = np.arange(0,len(sp.geoms))
    else:
        index_seeds_to_check = np.arange(0,len(sp.geoms),seed_count)
    print('nr of seeds to check: {}'.format(len(index_seeds_to_check)))

    # Create a list of seeds to check, based on the indices stored in the index list
    seeds_to_check = [seed for index,seed in enumerate(sp.geoms) if index in index_seeds_to_check]

    # Check the alignment for all seeds for horizontal and vertical lines separately
    hor_aligned_distances = [grid_hor.distance(seed) for seed in seeds_to_check]
    ver_aligned_distances = [grid_ver.distance(seed) for seed in seeds_to_check]

    hor_aligned = [dist<margin for dist in hor_aligned_distances]
    ver_aligned = [dist<margin for dist in ver_aligned_distances]

    # Result is only true if a seed is on both the vertical and horizontal lies
    results = [hor_aligned[i] and ver_aligned[i] for i in range(len(hor_aligned))]

    # Create a dataframe of the seeds that were checked for plotting purposes
    sp_df = gpd.GeoSeries(sp).get_coordinates().iloc[index_seeds_to_check]
    
    # Store the result in the dataframe for plotting
    sp_df['hor_distance'] = hor_aligned_distances
    sp_df['ver_distance'] = ver_aligned_distances
    sp_df['on_grid_ver'] = hor_aligned
    sp_df['on_grid_hor'] = ver_aligned
    sp_df['on_grid'] = results
    

    score = sum(results)/len(hor_aligned)

    return score, sp_df

def boustrophedon_decomposition(field,obstacles,base_ab,slope):
    """
    Decomposes the fields into subfields dependent on the position of the obstacle(s) and the orientation of the AB-lines.

    Parameters:
        field (geopandas Geoseries): Geoseries of Polygon object representing the field
        obstacles (list): list of GeoSeries of Polygons representing the obstacles in the field
        base_ab ()
        base_AB (numpy array): a line object that covers the entire y-range of the polygon with a direction specified in the edge attribute
        slope (float): slope of the base AB-line 

    Returns:
        decomposed_polygons (list): list of Polygon objects representing the subfields that comprise the field
    """
    field = field[0] # the outline of the field as a Polygon object
    step_size = 0.001
    minx,miny,maxx,maxy = field.bounds
    base_ab = LineString(base_ab)
    obstacles = [obstacle[0] for obstacle in obstacles] # obstacles in the field as Polygon objects
     
    # Create a sweep of possible cutting points
    if np.isnan(slope) == False:
        sweep_lines = [shapely.affinity.translate(base_ab,0,ytran) for ytran in np.arange(-miny,maxy,step_size)]
    else:
        sweep_lines = [shapely.affinity.translate(base_ab,xtran,0) for xtran in np.arange(-minx,maxx,step_size)]
    
    # Start with the original field, replace this later with the decomposed geometries
    decomposed_polygons = [field]
    # Loop over all the obstacles and keep track of the index
    for index_obs,obstacle in enumerate(obstacles):
        # Create an empty list of new decompositions for each loop
        new_decomposed_polygons = []
        # Loop over all the sweep lines
        for index,line in enumerate(sweep_lines):
            # Loop over all polygons in the list of decomposed polygons
            for poly in decomposed_polygons:
                # This if-statement makes sure that only the first line that touches an obstacle is used to cut it
                if index ==0 or index ==len(sweep_lines)-1:
                    if obstacle.intersects(line):
                        split_polys = split(poly, line)
                        for split_poly in split_polys.geoms:
                            new_decomposed_polygons.append(split_poly)
                else:
                    if sum([obstacle.intersects(line),obstacle.intersects(sweep_lines[index-1]) , obstacle.intersects(sweep_lines[index+1])]) ==2:
                        split_polys = split(poly, line)
                        for split_poly in split_polys.geoms:
                            new_decomposed_polygons.append(split_poly)
       
        # For each item in the new decomposed list, we check the intersection, if there is an intersection between 2 
        # Polygons, we add that intersection to the list of decomposed fields. We then check if any of the resulting decomposed
        # Fields fully contain another decomposed field, if that is the case, we remove the big field from the list
    
        for index_decomp, decomposed_polygon in enumerate(new_decomposed_polygons):
            # Calculate the intersection between 1 polygon and all others
            intersections = [decomposed_polygon.intersection(polygon) for polygon in new_decomposed_polygons] 
            # Check if the area of the intersection is >0
            intersections = [intersection for intersection in intersections if intersection.area > 0] 
            # Store the intersections in the new decomposition list
            new_decomposed_polygons.extend(intersections) 

            # This for loop removes any duplicate polygons:
            unique_polys = []
            for poly in new_decomposed_polygons:
                if not any(p.equals(poly) for p in unique_polys):
                    unique_polys.append(poly)
            new_decomposed_polygons = unique_polys

        # If it is the first iteration, we replace the decomposed polygons (which is just the field in the first iteration), with the newly decomposed polygons
        # In the other iterations, we extend it
        if index_obs == 0:
            decomposed_polygons = new_decomposed_polygons
        else:
            decomposed_polygons.extend(new_decomposed_polygons)
        
        # Here we substract the current obstacle from all of the polygons
        decomp_pols = []
        for index, polygon in enumerate(decomposed_polygons):
            for obstacle in obstacles:
                polygon = polygon.difference(obstacle)
            decomp_pols.append(polygon)

        decomposed_polygons = []
        for item in decomp_pols: # Convert the multipolygons created by the difference operation into a list of polygons
            if type(item) == shapely.MultiPolygon: # If it's a multipolygon, convert the geoms to a list and add it
                decomposed_polygons.extend(list(item.geoms))
            else:
                decomposed_polygons.append(item)
        
        # Remove the duplicates again (honestly no clue why that is necessary)
        unique_polys = []
        for poly in decomposed_polygons:
            if not any(p.equals(poly) for p in unique_polys):
                unique_polys.append(poly)
        decomposed_polygons = unique_polys

        to_remove = [] # Remove polygons that fully contain other polygons, doesn't remove them all for some reason 
        
        for index_decomp, decomposed_polygon in enumerate(decomposed_polygons):
            contains = [decomposed_polygon.contains(polygon) for polygon in decomposed_polygons]
            contains.pop(index_decomp)
            if True in contains:
                to_remove.append(index_decomp)
        for index in sorted(to_remove,reverse=True): 
            # Store the indices to remove in a list (to_remove) and then we remove these indices from the decomposed_polygons list
            del decomposed_polygons[index]

        fig, ax = plt.subplots(1,len(decomposed_polygons),figsize = (25,5)) # More plotting
        for i in range(len(decomposed_polygons)):
            x,y = decomposed_polygons[i].exterior.xy            
            ax[i].plot(x,y)
            ax[i].plot(*base_ab.xy, 'r--')
            x,y = field.exterior.xy
            ax[i].plot(x,y,c='r',alpha = 0.2)
        fig.suptitle('Final decomposition for chosen swath direction')
        plt.show()
 
    return decomposed_polygons

def pathplanning(data_path,include_obs,turning_rad,tractor_width,plotting,seed_distance,no_offset = False,scale:int=1):
    """
    main function for the pathplanning, loads the data and generates a path (maybe its better to make this a class but idk)
    
    args:
        data_path (str): filepath that contains the json file to be parsed
        include_obs (bool): boolean to indicate if you want to include the obstacles in the 
                            resulting field (will probably be usefull for some debugging)
        turning_rad (float): turning radius of the tractor in m
        tractor_width (float): tractor width (same as distance between swaths) in m
        plotting (bool): boolean to decide whether you want to plot the generated path or not
        seed_distance (float): distance between seeds,  same as the interpolation distance 

    returns:
        field (geopandas Geoseries): The polygon that defines the field
        best_path (pandas DataFrame): Dataframe containing the xy coordinates of each checkpoint of the path. 

    """
    if include_obs:
        field, obstacles = load_data(data_path,include_obs) 
        fig, ax = plt.subplots()
        field.plot(ax=ax, color='lightblue', edgecolor='black')
        plt.savefig("filter/path_plot.png")
    else:
        field = load_data(data_path,include_obs) 
        fig, ax = plt.subplots()
        field.plot(ax=ax, color='lightblue', edgecolor='black')
        plt.savefig("filter/path_plot.png")
    field_headlands, headland_size = generate_headlands(field,turning_rad,tractor_width)
    coordinates = field.get_coordinates()
    coordinates_headlands = field_headlands.get_coordinates()
    lines = edge_to_line(coordinates)

    # Initialize emtpy lists to contain the generated paths and path lengths
    paths = [] 
    commands = []
    path_lengths = []
    sp_list = []
    bases = []
    score_list = []
    sp_df_list = []
  
    for i in range(len(coordinates)-1): # Loop over all of the edges
        print('Finding path {}/{}'.format(i,len(coordinates)-1))
        try:
            # Calculate the basis AB line, use that to fill the field and clip the swaths
            line,slope = basis_AB_line(lines.iloc[i],coordinates_headlands)
            swath_list = fill_field_AB(line,slope,coordinates,tractor_width)
            swaths_clipped = clip_swaths(swath_list,field_headlands)
            base = find_ref(swaths_clipped,field_headlands,slope)
            bases.append(base)

            if include_obs:
                try:
                    decomposed_polygons = boustrophedon_decomposition(field_headlands,obstacles,line,slope)
                    print('Decomposition completed')
                except:
                    print('Decomposition failed')
                    continue
            else:
                decomposed_polygons = [field_headlands]
                print('No decomposition performed')
                
            # At this point, we have found a direction in which we can drive if there are no obstacles present. We now have to make a new field
            paths_subfields = []
            commands_subfields = []
            for index,decomposed_field in enumerate(decomposed_polygons):
                decomposed_field = gpd.GeoSeries(decomposed_field) # Convert decomposed polygons to a geoseries object (and change the name to work in the rest of the functions)
                swaths_clipped = clip_swaths(swath_list,decomposed_field)
                # Make a path with dubins curves
                path_new = generate_path(swaths_clipped,turning_rad)
                # Interpolate the path with some specified distance
                path_new, command_new = interpolate_path(path_new,seed_distance,base,no_offset)
                
                if index == 0:
                    paths_subfields = [path_new]
                    commands_subfields = [command_new]
                    path_old = path_new
                    continue

                connecting_curve = [gpd.GeoSeries(connect_paths(path_old[-1],path_new[0],turning_rad))]
                paths_subfields.append(connecting_curve)
                paths_subfields.append(path_new)
                commands_subfields.append(len(connecting_curve[0].get_coordinates())*[0])
                commands_subfields.append(command_new)
                path_old = path_new
                command_old = command_new

            paths_subfields = [val for sublist in paths_subfields for val in sublist]
            commands_subfields = [val for sublist in commands_subfields for val in sublist]
            # print(paths_subfields)
            path_df = path_to_df(paths_subfields,commands_subfields)
            headlands_path_connected = generate_connect_headlands(field,paths_subfields,tractor_width,turning_rad,headland_size,seed_distance)

            total_path = pd.concat([path_df,headlands_path_connected])
            total_path_no_dupl = total_path.drop_duplicates(['x','y'] ,ignore_index = True)
            paths.append(total_path_no_dupl)
            

            seed_count,sp= plantseeds(total_path_no_dupl,tractor_width,seed_distance)
            
            sp_list.append(sp)
            grid_hor,grid_ver,seed_count = create_grid(sp,seed_count,seed_distance,tractor_width)
            # grid_hor_prepped = shapely.prepared.prep(grid_hor)
            # grid_ver_prepped = shapely.prepared.prep(grid_ver)
            score,sp_df =  check_alignment(sp,grid_hor,grid_ver,seed_count,margin = 0.01,full_score = True)
            sp_df_list.append(sp_df)
            print('Path {}, score: {}'.format(i,score))
            score_list.append(score)
            commands.append(commands_subfields)


        except Exception as error: # Error handling to make sure the loop continues if an error pops up
            paths.append(None)
            commands.append(None)
        #     path_lengths.append(0)
            sp_list.append(None)
            sp_df_list.append(None)
            bases.append(None)
            score_list.append(0)
            print(error)
            
    # Finding the path with the best measure
    seeds_no = [len(gpd.GeoSeries(sp).get_coordinates()) for sp in sp_list] # total number of seeds for different paths
    best_path_index = np.argmax(np.array(score_list)) # for now the measure is total seed count
    best_score = score_list[best_path_index]
    best_path = paths[best_path_index]
    command = commands[best_path_index]

    sp = sp_list[best_path_index]
    sp_df = sp_df_list[best_path_index]
    sp_df['euclidian_dist'] = (sp_df['hor_distance']**2+sp_df['ver_distance']**2)**0.5
    base = bases[best_path_index]
    
    # Plot the path if it is specified
    if plotting:
        # Plotting the final (best) path
        fig, ax = plt.subplots()
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        field.plot(ax = ax,color = 'g',)
        field_headlands.boundary.plot(ax = ax,color = 'r')
        best_path.plot(x = 'x', y = 'y',ax = ax,color = 'magenta',marker = 'o',markersize = 1)
        ax.set_title('Final path')
        ax.get_legend().remove()
        plt.savefig('filter/final_path.png')



        # Plotting seed offsets for different directions 
        fig , ax2 = plt.subplots(1,3,figsize = (15,5))
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]
        viridis = matplotlib.colormaps['viridis']

        im1  = ax2[0].scatter(sp_df.x,sp_df.y, marker = 'o', c = sp_df['hor_distance'],s = 0.1,cmap = viridis)
        field.boundary.plot(ax = ax2[0],color = 'k')
        best_path.plot(x ='x',y='y',ax = ax2[0],color = 'm', alpha = 0.2)
        ax2[0].set_title('Distance in driving direction')

        im2 = ax2[1].scatter(sp_df.x,sp_df.y, marker = 'o', c = sp_df['ver_distance'],s = 0.1,cmap = viridis)
        best_path.plot(x ='x',y='y',ax = ax2[1],color = 'm', alpha = 0.2)
        field.boundary.plot(ax = ax2[1],color = 'k')
        ax2[1].set_title('Distance normal to driving direction')

        im3 = ax2[2].scatter(sp_df.x,sp_df.y, marker = 'o', c = sp_df['euclidian_dist'],s = 0.1,cmap = viridis)
        best_path.plot(x ='x',y='y',ax = ax2[2],color = 'm', alpha = 0.2)
        field.boundary.plot(ax = ax2[2],color = 'k')
        ax2[2].set_title('Euclidean distance')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.30, 0.01, 0.4])
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.set_label('Distance [m]')
        fig.suptitle('Seed positions for final path, total score: {}'.format(best_score))

        ax2[0].get_legend().remove()
        ax2[1].get_legend().remove()
        ax2[2].get_legend().remove()
        plt.savefig('filter/Seedoffsets.png')

        # Plotting seed alignment:
        _, ax = plt.subplots(1,3,figsize = (15,5))
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman"]        
        for (v,c) in [(True,'b'),(False,'r')]: # Colors based on whether a seed is on the grid or not
            ax[0].plot(sp_df.x[sp_df.on_grid_hor == v],sp_df.y[sp_df.on_grid_hor == v],'o',markersize = 1)
        for (v,c) in [(True,'b'),(False,'r')]:
            ax[1].plot(sp_df.x[sp_df.on_grid_ver == v],sp_df.y[sp_df.on_grid_ver == v],'o',markersize = 1)
        for (v,c) in [(True,'b'),(False,'r')]:
            ax[2].plot(sp_df.x[sp_df.on_grid == v],sp_df.y[sp_df.on_grid == v],'o',markersize = 1)

        ax[0].legend(['Aligned', 'Misaligned'])
        ax[1].legend(['Aligned', 'Misaligned'])
        ax[2].legend(['Aligned', 'Misaligned'])
        ax[0].set_title('horizontal alignment')
        ax[1].set_title('vertical alignment')
        ax[2].set_title('Total alignment')
        plt.savefig('filter/Seedalignment.png')

    
    return field,field_headlands,best_path,sp, swaths_clipped,base, total_path, bases,sp_df