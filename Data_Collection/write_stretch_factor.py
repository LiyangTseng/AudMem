import os
import librosa
import csv


def get_stretch_factor(relative_audio_path, request_length_in_sec=5):
    "strech variavle-length input clip to request length"
    
    y, sr = librosa.load(relative_audio_path)
    input_sample_num = y.shape[0]
    output_sample_num = sr*request_length_in_sec
    stretch_ratio = input_sample_num/output_sample_num
    
    return stretch_ratio

if __name__ == "__main__":
    
    subfolders = ["raw_variable_intro", "new_raw_variable_intro"]
    with open("stretch_factors.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file", "stretch_factor"])
        for subfolder in subfolders:
            files = os.listdir(os.path.join("Audios", subfolder))
            files = sorted(files, key=str.casefold)

            for filename in files:
                relative_audio_path = os.path.join("Audios", subfolder, filename)
                stretch_factor = get_stretch_factor(relative_audio_path)
                writer.writerow([relative_audio_path, stretch_factor])