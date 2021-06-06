import numpy as np
import pandas as pd
import datetime
import os
import json

isic_data_dir = '../../data/isic/' 
isic_descriptions_dir = os.path.join(isic_data_dir, 'Descriptions')
isic_images_dir = os.path.join(isic_data_dir, 'Images')


def read_row(data_instance, clinical_keys):
    
#     print(isic_images_dir)
#     print(data_instance['name'])
    
    image_name = data_instance['name']
   # print(np.where([x.startswith(image_name) for x in os.listdir(isic_images_dir)]))
    # need to get the extension
    this_list_idx = np.where([x.startswith(image_name) for x in os.listdir(isic_images_dir)])[0][0]
    image_fn = os.listdir(isic_images_dir)[this_list_idx]


    row_dict = {
        'study_name' : data_instance['dataset']['name'],
        'image_name': image_fn,
        'id': data_instance['_id']
    }
    try:
        row_dict['image_type'] =  data_instance['meta']['acquisition']['image_type']
    except:
        row_dict['image_type'] = np.nan

    
    for key in clinical_keys:
        try:
            row_dict[key] = data_instance['meta']['clinical'][key]
        except:
            row_dict[key] = np.nan

    return row_dict

def compile_instances(date_str_cutoff = '2019-01-01', save_newest_data = False):
    
    description_files = os.listdir(isic_descriptions_dir)
    print(len(description_files))
    
    with open(os.path.join(isic_descriptions_dir, description_files[0])) as f:
        data_0 = json.load(f)
        
    clinical_keys = list(data_0['meta']['clinical'].keys())
    
    df_keys = ['id','image_name', 'study_name', 'image_type'] + clinical_keys
    df = pd.DataFrame(columns=df_keys)
    
    if save_newest_data:
        df_new_data_included = pd.DataFrame(columns=df_keys)

    dt_cutoff = datetime.datetime.strptime(date_str_cutoff, '%Y-%m-%d').date()
    
    # takes at most a few minutes
    print('reading through description files')
    for i,description_file in enumerate(description_files):     
        if (i % 1000 == 0):
            print('{0}'.format(i))
        with open(os.path.join(isic_descriptions_dir, description_file)) as f:
            data_instance = json.load(f)
        
        date_time_str = data_instance['updated']
        date_added = datetime.datetime.strptime(date_time_str[:10], '%Y-%m-%d').date()

        if date_added < dt_cutoff:
            # only append to this df if the timestamp is before the cutoff
            instance_dict = read_row(data_instance, clinical_keys)
            df = df.append(instance_dict, ignore_index=True)
        
        if save_newest_data:
            # always append to this one if save_newest_data True
            # note: this will take longer
            instance_dict = read_row(data_instance, clinical_keys)
            df_new_data_included = df_new_data_included.append(instance_dict, 
                                                           ignore_index=True)

    out_dir = os.path.join(isic_data_dir,'isic_unprocessed.csv')
    print('saving data in ', out_dir)
    df.to_csv(out_dir, index=False)
    
    if save_newest_data:
        out_dir_all_data = os.path.join(isic_data_dir,'isic_unprocessed_new_data_included.csv')
        print('saving all data (including new data thats not in the Hidden stratification paper)',
              ' in ', out_dir_all_data)
        df_new_data_included.to_csv(out_dir_all_data, index=False)

if __name__ == "__main__":
    compile_instances()