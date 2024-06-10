import json
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
import geopandas as gpd
### The following script shows a quick example of how to load the geometry data from a JSON file, store it in a list and create geometric objects
### It also buffers the edges of the field and plots the original field and the field without headlands in the same figure


with open("./data/field_geometry/test_1.json") as json_file:
    json_data = json.load(json_file) # or geojson.load(json_file)

coordinates = []
for i in range(len(json_data['features'])):
    coordinates.append(json_data['features'][i]['geometry']['coordinates'])

polygon1 = gpd.GeoSeries(Polygon(coordinates[1][0]))
hole = gpd.GeoSeries(Polygon(coordinates[2][0]))
obstacle_buffer = hole.buffer(1)
polygon1_buffer = polygon1.buffer(-1)

field = polygon1.symmetric_difference(hole)

field_with_headlands = polygon1_buffer.symmetric_difference(obstacle_buffer)


fig, ax = plt.subplots()
field.plot(ax = ax, color = 'green')
field_with_headlands.boundary.plot(ax = ax,color = 'red')


plt.show()
