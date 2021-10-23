import os
import librosa
from tqdm import tqdm
import soundfile as sf


def strech_to_request_length(y, sr, request_length_in_sec=5):
    "strech variavle-length input clip to request length"
    input_sample_num = y.shape[0]
    output_sample_num = sr*request_length_in_sec
    strech_ratio = input_sample_num/output_sample_num
    streched_y = librosa.effects.time_stretch(y, strech_ratio)
    
    return streched_y

if __name__ == "__main__":
    # y, sr = librosa.load("Audios/new_raw_variable_intro/beforeNormalize_intro_WrRAZVJGImw.wav")
    # print(y.shape)
    # normalize_y = strech_to_request_length(y, sr, request_length_in_sec=5)
    # print(normalize_y.shape)
    # normalized_filename = "normalize_5s_intro_WrRAZVJGImw.wav"
    # sf.write(normalized_filename, normalize_y, sr)

    subfolders = ["raw_variable_intro", "new_raw_variable_intro"]
    for subfolder in subfolders:
            files = os.listdir(os.path.join("Audios", subfolder))
            if subfolder == "raw_variable_intro":
                output_folder = os.path.join("Audios", "normalized_intro")
            else:
                output_folder = os.path.join("Audios", "new_normalized_intro")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for filename in tqdm(files, desc="normalizing {}".format(subfolder)):
                relative_audio_path = os.path.join("Audios", subfolder, filename)
                y, sr = librosa.load(relative_audio_path)
                normalize_y = strech_to_request_length(y, sr, request_length_in_sec=5)
                output_filename = "normalize_5s_intro_" + filename[-15:]
                output_path = os.path.join(output_folder, output_filename)
                sf.write(output_path, normalize_y, sr)
