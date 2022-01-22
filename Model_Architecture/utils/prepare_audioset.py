import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd

def parse_csv(csv_path):
    ''' parse the csv and return the music-labeled ones from audio set as dataframe '''
    
    df = pd.read_csv(csv_path, sep=', ', header=None, skiprows=3) # skip first 3 rows of comments
    df.columns = ["YTID", "start_seconds", "end_seconds", "positive_labels"]
    music_id = "/m/04rlf" # via ../data/class_labels_indices.csv
    return df.loc[df["positive_labels"].str.contains(music_id)]
    
def download_wav(df, save_dir):
    ''' download audios by youtube-dl with given youtube ids '''
    
    os.makedirs(save_dir, exist_ok=True)

    for idx, row in df.iterrows():
        tmp_path = "{output_dir}/tmp".format(output_dir=save_dir)
        wav_path = "{output_dir}/{filename}.wav".format(output_dir=save_dir, filename="{:07d}".format(idx))
        if not os.path.exists(wav_path+".wav"):
            # download audio
            os.system("youtube-dl --extract-audio  --audio-format wav -o '{filename}.%(ext)s' {url}".format(
                            filename=tmp_path, url="https://www.youtube.com/watch?v={}".format(row["YTID"])))
            # truncate audio, ref: https://stackoverflow.com/questions/46508055/using-ffmpeg-to-cut-audio-from-to-position
            os.system("ffmpeg -ss {s_time} -i {in_file} -t 10 {out_file}".format(
                s_time=row["start_seconds"], in_file=tmp_path+".wav", out_file=wav_path
            ))
            os.system("rm {tmp}.wav".format(tmp=tmp_path))

if __name__ == "__main__":
    
    # !wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/unbalanced_train_segments.csv
    # !wget http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
    # !grep Music ../data/class_labels_indices.csv
    csv_path = "../data/unbalanced_train_segments.csv"
    save_dir = "../data/audioset/wav"

    print("parsing csv file, may take a while...")
    df = parse_csv(csv_path)
    print("downloading audios...")
    download_wav(df, save_dir)
    