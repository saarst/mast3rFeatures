import os
import json
from types import SimpleNamespace


def find_folders_at_depth(root_folder, target_depth):
    """Find all folders at the specified depth within the root folder."""
    folders_at_depth = set()

    for dirpath, dirnames, _ in os.walk(root_folder):
        # Calculate the depth of the current directory relative to the root folder
        if root_folder == dirpath:
            current_depth = 0
        else:
            current_depth = os.path.relpath(dirpath, root_folder).count(os.sep) + 1

        if current_depth == target_depth:
            folders_at_depth.add(dirpath)

    return sorted(folders_at_depth)

def load_arguments_from_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
        return json.loads(json.dumps(data), object_hook=lambda d: SimpleNamespace(**d))

def modify_json_field(file_path, field_key, new_value):
    # Open and read the JSON file
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Modify the specified field
    data[field_key] = new_value

    # Save the modified JSON back to the file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)



object_index_to_object_name = {
    0: 'alarm',
    1: 'backpack',
    2: 'bell',
    3: 'blocks',
    4: 'chicken',
    5: 'cream',
    6: 'elephant',
    7: 'grandfather',
    8: 'grandmother',
    9: 'hat',
    10: 'leather',
    11: 'lion',
    12: 'lunch_bag',
    13: 'mario',
    14: 'oil',
    15: 'school_bus1',
    16: 'school_bus2',
    17: 'shoe',
    18: 'shoe1',
    19: 'shoe2',
    20: 'shoe3',
    21: 'soap',
    22: 'sofa',
    23: 'sorter',
    24: 'sorting_board',
    25: 'stucking_cups',
    26: 'teapot',
    27: 'toaster',
    28: 'train',
    29: 'turtle'
}

object_name_to_object_index = {v: k for k, v in object_index_to_object_name.items()}

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp"}
