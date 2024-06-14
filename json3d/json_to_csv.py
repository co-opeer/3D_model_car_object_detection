import os
import json
import csv
import numpy as np
from collections import defaultdict

# Function to find the closest point to the center
def find_closest_point(group):
    # Flatten all points into a single list
    all_points = np.array([point for shape in group for point in shape['points']])

    # Calculate mean point
    mean_point = np.mean(all_points, axis=0)

    # Calculate distances from the mean point to all points
    distances = np.linalg.norm(all_points - mean_point, axis=1)

    # Find the index of the closest point
    closest_point_index = np.argmin(distances)

    # Get the coordinates of the closest point
    closest_point = all_points[closest_point_index]

    return closest_point

# Function to process shapes
def process_shapes(shapes, image_path):
    grouped_shapes = defaultdict(list)
    for shape in shapes:
        label = shape['label']
        grouped_shapes[label].append(shape)

    results = []
    for label, group in grouped_shapes.items():
        if len(group) != 3:
            continue

        # Find the central point
        central_point = find_closest_point(group)

        # Separate the dimensions
        dimensions = []
        for shape in group:
            start, end = shape['points']
            if abs(start[0] - central_point[0]) > 5 or abs(start[1] - central_point[1]) > 5:
                dimensions.append(start)
            elif abs(end[0] - central_point[0]) > 5 or abs(end[1] - central_point[1]) > 5:
                dimensions.append(end)

        # Assigning width, height, and length based on sorted distances
        (xwidth, ywidth) = dimensions[0]
        (xheight, yheight) = dimensions[1]
        (xlength, ylength) = dimensions[2]

        xcenter, ycenter = central_point

        results.append({
            'image': image_path,
            'xcenter': xcenter,
            'ycenter': ycenter,
            'xwidth': xwidth,
            'ywidth': ywidth,
            'xheight': xheight,
            'yheight': yheight,
            'xlength': xlength,
            'ylength': ylength
        })

    return results

# Function to process all JSON files in a folder
def process_folder(folder_path):
    all_results = []
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        with open(json_path) as f:
            data = json.load(f)
        shapes = data['shapes']
        image_path = data['imagePath']
        results = process_shapes(shapes, image_path)
        all_results.extend(results)
    return all_results

# Define folder path containing JSON files
folder_path = r"C:\Users\PC\OneDrive\Робочий стіл\photo for dataset\cars"

# Process all JSON files in the folder
results = process_folder(folder_path)

# Define CSV output file path
csvfile = r'C:\Users\PC\PycharmProjects\3D_model_car_object_detection\json3d\output.csv'

# Write results to CSV file
with open(csvfile, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=['image', 'xcenter', 'ycenter', 'xwidth', 'ywidth', 'xheight', 'yheight',
                                              'xlength', 'ylength'])
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print("CSV file has been created successfully.")
