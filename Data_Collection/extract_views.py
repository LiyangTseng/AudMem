import pandas as pd

def extract_views(df, output_path):
    ''' extract views from intro_locations.csv and write to another csv'''

    df = df[["file", "views"]]
    for idx in range(len(df)):
        df.loc[idx, "file"] = "normalize_5s_intro_" + df.loc[idx, "file"].split('/')[-1][:11] + ".wav"
        
    df = df.rename(columns={'file': 'track'})
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    df = pd.read_csv("intro_locations.csv")
    extract_views(df, "track_popularity.csv") 
