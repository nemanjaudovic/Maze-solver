import numpy as np
from PIL import Image
# path = input()
import heapq

import imageio
from PIL import Image


def image_to_graph(image_array):
    height, width = image_array.shape

    # Initialize an empty dictionary to store the graph
    graph = {}

    # Helper function to convert 2D coordinates to 1D index
    def coord_to_index(y, x):
        return y * width + x

    for y in range(height):
        for x in range(width):
            # If the current pixel is white (represented by 255), check its neighbors
            if image_array[y, x] == 255:
                # Initialize the list of neighbors for the current pixel
                neighbors = []
                # Check the 4 neighboring pixels (excluding diagonals)
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        # Skip diagonal neighbors and the current pixel
                        if abs(dy) + abs(dx) != 1:
                            continue
                        # Calculate the coordinates of the neighboring pixel
                        ny, nx = y + dy, x + dx
                        # Check if the neighboring pixel is within the image boundaries
                        if 0 <= ny < height and 0 <= nx < width:
                            # If the neighboring pixel is also white, add it to the list of neighbors
                            if image_array[ny, nx] == 255:
                                neighbors.append(coord_to_index(ny, nx))
                # Add the list of neighbors to the graph dictionary for the current pixel
                graph[coord_to_index(y, x)] = neighbors

    return graph

# def dijkstra(graph, start_node):
#     # Initialize distance array with infinity for all nodes
#     distances = {node: np.inf for node in graph}
#     # Distance from start node to itself is 0
#     distances[start_node] = 0
#
#     # Initialize array to keep track of visited nodes
#     visited = {node: False for node in graph}
#
#     # Dijkstra's algorithm
#     while True:
#         # Find the node with the shortest distance from the start node
#         min_dist = np.inf
#         min_node = None
#         for node in graph:
#             if not visited[node] and distances[node] < min_dist:
#                 min_dist = distances[node]
#                 min_node = node
#
#         if min_node is None:
#             break
#
#         # Mark the selected node as visited
#         visited[min_node] = True
#
#         # Update distances for neighboring nodes
#         for neighbor in graph[min_node]:
#             if not visited[neighbor]:
#                 new_distance = distances[min_node] + 1  # Assuming all edges have weight 1
#                 if new_distance < distances[neighbor]:
#                     distances[neighbor] = new_distance
#
#     return distances
def dijkstra(graph, start_node):
    distances = {node: np.inf for node in graph}
    distances[start_node] = 0
    visited = set()
    heap = [(0, start_node)]

    while heap:
        dist, node = heapq.heappop(heap)
        if node in visited:
            continue

        visited.add(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                new_distance = dist + 1
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))

    return distances
def shortest_distance_between_nodes(matrix, start_node, end_node):
    # Apply Dijkstra's algorithm to find the shortest distances from the start node
    shortest_distances = dijkstra(matrix, start_node)
    # Return the shortest distance from the start node to the end node
    return shortest_distances[end_node]


def coord_in_num(array,i,j):
    return i*array.shape[1] + j




image_path = input() #Path to image



# Read the image
image = imageio.imread(image_path)

# Convert the image to grayscale
gray_image = image.mean(axis=2, dtype=int)
array = np.array(gray_image)
background_color = array[0][0]
entrances = []
graph = image_to_graph(array)
num_of_teleports=input() #Number of teleports - 0 if there arent any
num_of_teleports = int(num_of_teleports)
teleport_coords=[]
for i in range(num_of_teleports):
    input_line = input()
    numbers=input_line.split()
    teleport_coords.append((numbers[0],numbers[1]))
k=0
#Ccalculating coordinates of entrencies
for i in range(1, array.shape[0] - 1):
    if array[i][0] != array[i-1][0]:
        k +=1
        if array[i][0] == 255:
            entrances.append((i,0))
        else:
            entrances.append((i-1, 0))

for i in range(1, array.shape[0] - 1):
    if array[i][array.shape[1]-1] != array[i-1][array.shape[1]-1]:
        k += 1
        if array[i][array.shape[1]-1] == 255:
            entrances.append((i, array.shape[1]-1))
        else:
            entrances.append((i - 1, array.shape[1]-1))


for i in range(1, array.shape[1] - 1):
    if array[0][i] != array[0][i-1]:
        k +=1
        if array[0][i] == 255:
            entrances.append((0,i))
        else:
            entrances.append((0,i-1))
for i in range(1, array.shape[1] - 1):
    if array[array.shape[0] - 1][i] != array[array.shape[0] - 1][i-1]:
        k += 1
        if array[array.shape[0] - 1][i] == 255:
            entrances.append((array.shape[0] - 1,i))
        else:
            entrances.append((array.shape[0] - 1,i-1))
k = int(k/2)
print(k)
if k !=0:
    #Making list of tuples inside which are coordinates of entrencies
    ent=[]
    for i in range(int(len(entrances)/2)):
        ent.append((entrances[2*i],entrances[2*i+1]))
    coords_entries=ent
    min_dist=np.inf
    for i in range(len(coords_entries)-1):
        for j in range(i+1, len(coords_entries)):
            a = shortest_distance_between_nodes(graph, coord_in_num(array, coords_entries[i][0][0], coords_entries[i][0][1]), coord_in_num(array, coords_entries[j][0][0], coords_entries[j][0][1]))
            b= shortest_distance_between_nodes(graph, coord_in_num(array, coords_entries[i][1][0], coords_entries[i][1][1]), coord_in_num(array, coords_entries[j][0][0], coords_entries[j][0][1]))
            c = shortest_distance_between_nodes(graph, coord_in_num(array, coords_entries[i][0][0], coords_entries[i][0][1]), coord_in_num(array, coords_entries[j][1][0], coords_entries[j][1][1]))
            d = shortest_distance_between_nodes(graph, coord_in_num(array, coords_entries[i][1][0], coords_entries[i][1][1]), coord_in_num(array, coords_entries[j][1][0], coords_entries[j][1][1]))
            new=min(a,b,c,d)
            if new<min_dist:
                min_dist=new
    print(int(min_dist)+1)
    if num_of_teleports !=0:
        for i in range(len(num_of_teleports)):
            graph[num_of_teleports[i][0]][num_of_teleports[i][1]] = graph[num_of_teleports[i][1]][num_of_teleports[i][0]] = 1
        min_dist = np.inf
        for i in range(len(coords_entries) - 1):
            for j in range(i + 1, len(coords_entries)):
                a = shortest_distance_between_nodes(graph,
                                                    coord_in_num(array, coords_entries[i][0][0], coords_entries[i][0][1]),
                                                    coord_in_num(array, coords_entries[j][0][0], coords_entries[j][0][1]))
                b = shortest_distance_between_nodes(graph,
                                                    coord_in_num(array, coords_entries[i][1][0], coords_entries[i][1][1]),
                                                    coord_in_num(array, coords_entries[j][0][0], coords_entries[j][0][1]))
                c = shortest_distance_between_nodes(graph,
                                                    coord_in_num(array, coords_entries[i][0][0], coords_entries[i][0][1]),
                                                    coord_in_num(array, coords_entries[j][1][0], coords_entries[j][1][1]))
                d = shortest_distance_between_nodes(graph,
                                                    coord_in_num(array, coords_entries[i][1][0], coords_entries[i][1][1]),
                                                    coord_in_num(array, coords_entries[j][1][0], coords_entries[j][1][1]))
                new = min(a, b, c, d)
                if new < min_dist:
                    min_dist = new
        print(int(min_dist) + 1)
    else:
        print(int(min_dist) + 1)
else:
    print('0')
    print('0')