import pickle
import pandas as pd
import os
import shutil
import json


# open dataframes from pickled file 
df_kenya = pd.read_pickle('kenya_information.pkl')
file_names = df_kenya['file_name'].to_list()
instance_file = 'iwildcam2022_mdv4_detections.json'

def filter_background_ims(dataframe_locs, dataframe_masks, location_list):
   '''Given a dataframe of the locations corresponding to files, a masks dataframe (the dataframe with all the labels) and a location list of cameras,
   it returns a dataframe where only 10 percent of the images from your defined location list which contain only a background are left. This tackles the skewed dataset.'''

    # first make list of all images at defined locations
   sub_ims = dataframe_locs[dataframe_locs['location'].isin(location_list)]
   filenames = list(sub_ims['file_name'])

   # filter on useful files
   loc_frame =  dataframe_masks[dataframe_masks['file'].isin(filenames)]
   loc_frame = loc_frame[loc_frame['max_detection_conf']< 0.5] # leaves empty images
   loc_frame = loc_frame.sample(frac=1).reset_index(drop=True)
   # take 90 % of those images and drop them
   num_columns = int(len(loc_frame.columns) * 0.9)

    # Take the first 90% of the columns
   loc_frame = loc_frame.iloc[:, :num_columns]

   dataframe = dataframe_masks[~dataframe_masks.isin(loc_frame)].dropna()
   return dataframe

def generate_image_folder(file_names, target_folder_name):
    '''Creates folder and adds all filenames in the folder.'''
    try:
        current_folder = os.getcwd()
        os.path.join(current_folder, target_folder_name)
        os.mkdir(os.path.join(current_folder, target_folder_name))
    except:
        pass
    for source in file_names:
        file = os.path.join('test',source)
        shutil.copy(file, target_folder_name)


def generate_im_masks(filename,folder):
    '''Create a dataframe from the image masks'''
    data = json.load(open(filename))
    bounding_frame = pd.DataFrame(data['images'])

    # list all files in the folder
    current_folder = os.getcwd()
    file_names = os.listdir(os.path.join(current_folder, folder))

    # change inconistent naming convention
    names = []
    for i, name in enumerate(file_names):
        names.append('test/'+name)

    subset = bounding_frame[bounding_frame['file'].isin(names)]
    name_col = subset['file'].tolist()
    for i, name in enumerate(name_col):
        name_col[i] = name[5:]



    subset['file'] = name_col
    # save file
    subset.to_csv('kenya_labels.csv')
    return subset


def filter_ims(file, min_conf = 0.01, max_conf = .9):
    '''Filter out all images with uncertainties in labels.'''

    df = pd.read_csv(file)

    df_filtered = df[(min_conf>=df['max_detection_conf']) | (max_conf<=df['max_detection_conf'])]

    return df_filtered

  
# generate_im_masks(instance_file, 'kenya_ims')
filtered = filter_ims('kenya_labels.csv')
final_df = filter_background_ims(df_kenya,filtered,[430,150])
# final_df.to_csv('filtered_kenya_labels.csv')