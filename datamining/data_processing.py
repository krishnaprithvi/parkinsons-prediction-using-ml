import pandas as pd
import zipfile
import os

def load_data_from_zip(zip_file_path):
    dataframes = {}
    with zipfile.ZipFile(zip_file_path, 'r') as z:
        for file_name in z.namelist():
            if file_name.endswith('.data'):
                with z.open(file_name) as f:
                    dataframes[file_name] = pd.read_csv(f)
            elif file_name.endswith('.names'):
                with z.open(file_name) as f:
                    dataframes[file_name] = f.read().decode('utf-8')
    return dataframes

def preprocess_parkinsons_data(parkinsons_df):
    if 'name' in parkinsons_df.columns:
        parkinsons_df.drop(columns=['name'], inplace=True)
    # Save processed data
    processed_dir = 'data/processed'
    os.makedirs(processed_dir, exist_ok=True)
    parkinsons_df.to_csv(os.path.join(processed_dir, 'parkinsons_processed.csv'), index=False)
    return parkinsons_df

def split_and_scale(parkinsons_df):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    X = parkinsons_df.drop(columns=['status'])
    y = parkinsons_df['status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
