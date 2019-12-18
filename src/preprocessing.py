from sklearn.preprocessing import StandardScaler
import argparse
import pandas as pd
import numpy as np
import os.path

def convert_symbol_to_label(sequence):
    label_type = sequence.split("_")[-1]
    
    if label_type == "CTRL":
        return 0
    
    elif label_type == "DHT":
        return 1
    
    elif label_type == "P4":
        return 2
    else:
        return -1
    
def normalize_data_features(data):
    normalized_data = data.copy()
    normalizer = StandardScaler()
    data_array = normalizer.fit_transform(normalized_data[normalized_data.columns.difference(['label', 'full_label'])])
    normalized_data = pd.DataFrame(np.column_stack((normalized_data["full_label"].values,data_array,normalized_data["label"].values)),
                                   columns = normalized_data.columns).set_index(normalized_data.index)
    return normalized_data

def normalize_data_samples(data):
    normalized_data = data.copy()
    normalizer = StandardScaler()
    data_array = normalizer.fit_transform(normalized_data[normalized_data.columns.difference(['label', 'full_label'])].T).T
    normalized_data = pd.DataFrame(np.column_stack((normalized_data["full_label"].values,data_array, normalized_data["label"].values)),
                                   columns = normalized_data.columns).set_index(normalized_data.index)
    return normalized_data

def preprocess_dataset(args):
    """
    Generate dataset with features in columns and samples in rows from the raw dataset.
    
    Parameters
    ----------
    args
        dataset_path: str
            Path to the raw_train.csv dataset.
        saving_path: str
            Path to the generated train.csv dataset.  
    """
    # Load cvs dataset
    if os.path.isfile(args['dataset_path']):
        df_clean_data = pd.read_csv(args['dataset_path'], sep=",")
    else:
        print ("File not exist : {}".format(args['dataset_path']))
    
    if args["norm"] == "features":
        df_clean_data = normalize_data_features(df_clean_data)
        file_name = "normalized_features_train.csv"
        
    elif args["norm"] == "samples":
        df_clean_data = normalize_data_samples(df_clean_data)
        file_name = "normalized_samples_train.csv"
    else:
        file_name = "normalized_train.csv"
    
    df_clean_data.to_csv(args['saving_path']+file_name, index=False)
    print("Cleaned dataset saved at: ", args['saving_path']+file_name)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--norm', type=str, default="none",  help='Type of normalization: features, samples, none')
    parser.add_argument('--dataset_path', type=str, default="../data/real_raw_train.csv", help='Path of the dataset CSV file')
    parser.add_argument('--saving_path', type=str, default="../data/", help='Path where you want to save preprocessed dataset CSV file')
    args = vars(parser.parse_args())

    preprocess_dataset(args)