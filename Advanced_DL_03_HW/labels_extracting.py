import json
import os

# Input JSON annotation file and output directory
json_path = './football/Match_1951_1_0_subclip/Match_1951_1_0_subclip.json'
output_dir = './football/Match_1951_1_0_subclip/labels'

with open(json_path, 'r') as json_file:
    data = json.load(json_file)

os.makedirs(output_dir, exist_ok=True)

# Extracting information to the .txt file
def extract_image(image_data):
    image_id = image_data['id']
    width = image_data['width']
    height = image_data['height']

    # Save the annotations
    annotations = []

    for annotation in data['annotations']:
        if annotation['image_id'] == image_id:
            category_id = annotation['category_id']
            bbox = annotation['bbox']
            x, y, w, h = bbox
            # Convert into normalized coordinates
            x_norm = x / width
            y_norm = y / height
            w_norm = w / width
            h_norm = h / height

            # Form the annotation string
            annotation_str = f'{category_id} {x_norm} {y_norm} {w_norm} {h_norm}'
            annotations.append(annotation_str)

    annotations_str = '\n'.join(annotations)
    output_file_path = os.path.join(output_dir, f'frame_{image_id:02d}.txt')
    with open(output_file_path, 'w') as txt_file:
        txt_file.write(annotations_str)

# Loop through the images and create annotation files for each frame
for image_data in data['images']:
    extract_image(image_data)
