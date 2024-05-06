import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
import copy
import cv2
import os
import shutil
import json


def predict_field(image, conf_model):
    my_model = YOLO("best.pt")
    results = list(
        my_model(image, conf=conf_model, save_json=True)
    )  # If dont have the rewrite predict.py please deleate save_json=True
    result = results[0]

    # Convert PyTorch tensor to NumPy array
    mask_array = result.masks.data.cpu().numpy()

    # Create a blank image to store all mask information
    combined_mask = Image.new(
        "RGB", (mask_array.shape[2], mask_array.shape[1]), color=(255, 255, 255)
    )

    # Iterate through each mask
    for idx, mask in enumerate(mask_array):
        # Convert mask to PIL image
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")

        # Assign a color to each mask (for simplicity, random colors are used here)
        color = tuple([int(c) for c in np.random.randint(0, 256, size=3)])

        # Draw the mask on the combined image using the mask's pixel values
        draw = ImageDraw.Draw(combined_mask)
        draw.bitmap((0, 0), mask_image, fill=color)

    # Display the combined mask image
    os.makedirs("test", exist_ok=True)
    combined_mask.save("test/test1.png")
    return combined_mask
    # combined_mask.show()


def predict_json(scale,approx):
    # Path: Directory containing segmentation masks
    # Using relative path
    path = "test"
    files = os.listdir(path)

    # Iterate through each file in the directory
    for file in files:
        name = file.split(".")[0]
        file_path = os.path.join(path, name + ".png")
        img = cv2.imread(file_path)
        H, W = img.shape[0:2]  # Get the height and width of the image
        #print(H, W)

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply Otsu's thresholding to get a binary image
        ret, bin_img = cv2.threshold(
            gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        # Find contours in the binary image
        cnt, hit = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

        # Open a text file to write the contour coordinates
        # Using relative path
        f = open("test/{}.json".format(file.split(".")[0]), "w+")
        f.write("{\n")
        f.write('  "type": "FeatureCollection",\n')
        f.write('  "features": [\n')

        for index, field in enumerate(cnt):
            # Choose epsilon (approximation accuracy). You can adjust this value.
            epsilon = approx * cv2.arcLength(field, True)  # 2% of the arc length

            # Approximate the contour to create a smoother polygon
            approx_polygon = cv2.approxPolyDP(field, epsilon, True)
            result = []
            for i in approx_polygon:
                temp = list(i[0])
                temp[0] /= W  # Normalize x-coordinate
                temp[1] /= H  # Normalize y-coordinate
                result.append(temp)

            # If the contour is not empty, save it as a closed polygon
            if result:
                result.append(result[0])  # Add the starting point to close the polygon

                # Write the contour coordinates into the JSON file
                f.write("    {\n")
                f.write('      "type": "Feature",\n')
                f.write('      "properties": {\n')
                f.write('        "Name": "field{}"\n'.format(index))
                f.write("      },\n")
                f.write('      "geometry": {\n')
                f.write('        "type": "Polygon",\n')
                f.write('        "coordinates": [\n')
                f.write("          [\n")

                for idx, line in enumerate(result):
                    f.write("            [\n")
                    f.write("              {},\n".format(line[0]*scale))
                    f.write("              {}\n".format(line[1]*scale))
                    f.write(
                        "            ]{}\n".format("," if idx < len(result) - 1 else "")
                    )

                f.write("          ]\n")
                f.write("        ]\n")
                f.write("      }\n")
                if index < len(cnt) - 1:
                    f.write("    },\n")
                else:
                    f.write("    }\n")

        f.write("  ]\n")
        f.write("}\n")
        f.close()


def filter_json(field:int):
    with open("test/test1.json", "r") as f:
        geojson_data = json.load(f)

    # Extract the "features" list
    features = geojson_data["features"]

    # Filter the features to only include those with the "Name" property equal to "contour1"
    contour1_features = [
        feature for feature in features if feature["properties"].get("Name") == f"field{str(field)}"
    ]

    # Create a new GeoJSON with the filtered features
    filtered_geojson = {"type": "FeatureCollection", "features": contour1_features}

    # Save the new GeoJSON to a file
    os.makedirs("filter", exist_ok=True)
    with open("filter/filtered_geojson.json", "w") as f:
        json.dump(filtered_geojson, f, indent=2)
