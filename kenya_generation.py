import pickle
import pandas as pd
import os
import shutil


df_kenya = pd.read_pickle('kenya_information.pkl')
file_names = df_kenya['file_name'].to_list()
instance_file = 'iwildcam2022_mdv4_detections.json'
print(df_kenya.info())

def filter_background_ims(dataframe_locs, dataframe_masks, location_list):
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
#    loc_frame = loc_frame[loc_frame.index()<len(loc_frame.columns)*0.9]
   print(loc_frame.info())

   dataframe = dataframe_masks[~dataframe_masks.isin(loc_frame)].dropna()
   print(dataframe.info())
   return dataframe






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


def filter_ims(file, min_conf = 0.01, max_conf = .9):
    '''Filter out all images with uncertainties in labels.'''

    df = pd.read_csv(file)
    # print('original:')
    # print(df.info())
    df_filtered = df[(min_conf>=df['max_detection_conf']) | (max_conf<=df['max_detection_conf'])]
    # print('filtered')
    # print(df_filtered.info())
    return df_filtered

  
# generate_im_masks(instance_file, 'kenya_ims')
filtered = filter_ims('kenya_labels.csv')
final_df = filter_background_ims(df_kenya,filtered,[430,150])
final_df.to_csv('filtered_kenya_labels.csv')