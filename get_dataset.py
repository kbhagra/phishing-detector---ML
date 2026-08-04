import os 
import shutil 
import kagglehub 
import pandas as pd 
def load_dataset (dest_dir ="data"):
    """
    Downloads the 'phishing-dataset-for-machine-learning' dataset using kagglehub
    and copies the CSV file to a local destination directory.
    """
    print ("Downloading dataset via kagglehub...")
    download_path =kagglehub .dataset_download ("shashwatwork/phishing-dataset-for-machine-learning")
    print (f"Downloaded raw dataset files to cache: {download_path }")
    os .makedirs (dest_dir ,exist_ok =True )
    csv_name ="Phishing_Legitimate_full.csv"
    cached_csv_path =os .path .join (download_path ,csv_name )
    local_csv_path =os .path .join (dest_dir ,csv_name )
    if os .path .exists (cached_csv_path ):
        shutil .copy (cached_csv_path ,local_csv_path )
        print (f"Copied dataset to local path: {local_csv_path }")
    else :
        for root ,_ ,files in os .walk (download_path ):
            for file in files :
                if file .endswith (".csv"):
                    cached_csv_path =os .path .join (root ,file )
                    local_csv_path =os .path .join (dest_dir ,file )
                    shutil .copy (cached_csv_path ,local_csv_path )
                    print (f"Copied {file } to local path: {local_csv_path }")
                    break 
    df =pd .read_csv (local_csv_path )
    return df ,local_csv_path 
if __name__ =="__main__":
    df ,csv_path =load_dataset ()
    print ("\n--- Dataset Summary ---")
    print (f"Dataset shape: {df .shape } (rows, columns)")
    print ("\nFirst 5 rows:")
    print (df .head ())
    print ("\nColumns list:")
    print (df .columns .tolist ())
