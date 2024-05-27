import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
import copy
import cv2
import os
import shutil
import json
import numpy as np

def predict_field(image, conf_model):
    # Display the combined mask image
    os.makedirs("filter", exist_ok=True)
    my_model = YOLO("best.pt")
    results = list(
        my_model(image, conf=conf_model, save_json=True)
    )  # If dont have the rewrite predict.py please deleate save_json=True
    result = results[0]
    if result.masks is None:
        image = np.ones((640,640))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,"No prediction",org = (200, 320),fontFace=font ,fontScale=1,color=(255,0,0),thickness=2,lineType=cv2.LINE_AA)
        cv2.imwrite("test/test1.png",image)
    
    else:
        # Convert PyTorch tensor to NumPy array
        mask_array = result.masks.data.cpu().numpy()

        # Create a blank image to store all mask information
        combined_mask = Image.new(
            "RGB", (mask_array.shape[2], mask_array.shape[1]), color=(255, 255, 255)
        )
        print("Mask size: ",combined_mask.size)
        combined_mask.save("filter/mask_image.png")
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


def predict_json(image,scale,approx):
    img = cv2.imread(image)
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
    f = open("test/test1.json", "w+")
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
            temp[0] /= 1  # Can be used to normalize the pixels. But is deprecated because of the scaling
            temp[1] /= 1  # Can be used to normalize the pixels. But is deprecated because of the scaling
            result.append(temp)

        # If the contour is not empty, save it as a closed polygon
        if len(result)>4:
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
    try:
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
    except:  # noqa: E722
        print("Now field with this number")

def make_bridge(image,pixel1:int=1,pixel2:int=1,thickness_bridge:int=10):
    # Read the image
    image = cv2.imread(image)

    # Check if the image was successfully read
    if image is None:
        raise ValueError("Unable to read the image")

     # Convert the image to grayscale
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Otsu's thresholding to get a binary image
    _, binary = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    # Invert the binary image
    binary = cv2.bitwise_not(binary)

    # Display the binary image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image (Inverted)')
    plt.savefig("filter/bridge_bitwise_image.png")

    # Detect contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    # Draw a line connecting the centers of the two polygons
    cv2.line(binary, pixel1, pixel2, 255, thickness=thickness_bridge)  # Adjust the thickness of the connecting line

    # Display the final connected mask using matplotlib
    cv2.imwrite("filter/connected_bridge.png",binary)