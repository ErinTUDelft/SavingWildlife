import pickle
import pandas as pd
import os
import shutil


df_kenya = pd.read_pickle('kenya_information.pkl')
file_names = df_kenya['file_name'].to_list()
instance_file = 'iwildcam2022_mdv4_detections.json'


def generate_image_folder(file_names, target_folder_name):
    try:
        current_folder = os.getcwd()
        os.path.join(current_folder, target_folder_name)
        os.mkdir(os.path.join(current_folder, target_folder_name))
    except:
        pass
    for source in file_names:
        file = os.path.join('test',source)
        shutil.copy(file, target_folder_name)


# generate_image_folder(file_names, 'kenya_ims')
import json
def generate_im_masks(filename,folder):
    data = json.load(open(filename))
    bounding_frame = pd.DataFrame(data['images'])

    # list all files in the folder
    current_folder = os.getcwd()
    file_names = os.listdir(os.path.join(current_folder, folder))

    # print(names)
    names = []
    for i, name in enumerate(file_names):
        names.append('test/'+name)

    # bounding_frame['file'] = file_names
    # print(names[1])
    subset = bounding_frame[bounding_frame['file'].isin(names)]
    name_col = subset['file'].tolist()
    for i, name in enumerate(name_col):
        name_col[i] = name[5:]

        # print(name_col[i])

    subset['file'] = name_col
    print(subset.info())
    subset.to_csv('kenya_labels.csv')
    return subset



    
generate_im_masks(instance_file, 'kenya_ims')