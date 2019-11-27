import pandas as pd

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

def generate_train_dataset(raw_train_path, clean_train_path):
    """
    Generate dataset with features in columns and samples in rows from the raw dataset.
    
    Parameters
    ----------
    raw_train_path: str
        Path to the raw_train.csv dataset.
    clean_train_path: str
        Path to the generated train.csv dataset.  
    """
    # Load cvs dataset
    df_raw_data = pd.read_csv(raw_train_path, sep=";")
    
    # Transpose the dataset
    df_clean_data = df_raw_data.T
    
    # Set the gene symbols as column index
    df_clean_data.columns = df_clean_data.iloc[1]
    
    # Clean unrelevant rows (ensgene, symbol, biotype, description)
    df_clean_data = df_clean_data.drop(df_clean_data.index[[0,1,2,3]])
    
    # Reset index and create new column with full label names
    df_clean_data = df_clean_data.rename_axis('full_label').reset_index()
    
    # Create labels column
    df_clean_data["label"] = df_clean_data["full_label"].apply(convert_symbol_to_label)
    
    df_clean_data.to_csv(clean_train_path, index=False)
    print("Cleaned dataset saved at: ", clean_train_path)
    
    
if __name__ == "__main__":
    raw_train_path = "../data/raw_train.csv"
    clean_train_path = "../data/train.csv"

    generate_train_dataset(raw_train_path, clean_train_path)