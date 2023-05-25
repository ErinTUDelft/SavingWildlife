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

def generate_im_masks(filename):
    df = pd.read_json(filename)
    print(df.head())

generate_im_masks(instance_file)