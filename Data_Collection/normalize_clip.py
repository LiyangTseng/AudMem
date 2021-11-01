import os
import librosa
from tqdm import tqdm
import soundfile as sf
from pydub import AudioSegment

LENGTH = 5

def strech_to_request_length(y, sr, request_length_in_sec=5):
    "strech variavle-length input clip to request length"
    input_sample_num = y.shape[0]
    output_sample_num = sr*request_length_in_sec
    strech_ratio = input_sample_num/output_sample_num
    streched_y = librosa.effects.time_stretch(y, strech_ratio)
    
    return streched_y

def normalize_volumn(sound, target_loudness):
    "normalize all audios to same loudness"
    change_in_dBFS = target_loudness - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

if __name__ == "__main__":

    audio_dir = "Audios/235_normalized"
    output_dir = audio_dir+"_volumn_normalized"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    audio_files = os.listdir(audio_dir)
    for audio_file in tqdm(audio_files):
        y = AudioSegment.from_file(os.path.join(audio_dir, audio_file), "wav")
        normalized_y = normalize_volumn(y, -20)
        normalized_y.export(os.path.join(output_dir, audio_file), format="wav")

    # subfolders = ["raw_variable_intro", "new_raw_variable_intro"]
    # for subfolder in subfolders:
    #         files = os.listdir(os.path.join("Audios", subfolder))
    #         if subfolder == "raw_variable_intro":
    #             output_folder = os.path.join("Audios", "{}_normalized_intro".format(LENGTH))
    #         else:
    #             output_folder = os.path.join("Audios", "{}_new_normalized_intro".format(LENGTH))
    #         if not os.path.exists(output_folder):
    #             os.makedirs(output_folder)

    #         for filename in tqdm(files, desc="normalizing {}".format(subfolder)):
    #             relative_audio_path = os.path.join("Audios", subfolder, filename)
    #             y, sr = librosa.load(relative_audio_path, sr=48000)
    #             normalized_y = strech_to_request_length(y, sr, request_length_in_sec=LENGTH)
    #             output_filename = "normalize_{}s_intro_".format(LENGTH) + filename[-15:]
    #             output_path = os.path.join(output_folder, output_filename)
    #             sf.write(output_path, normalized_y, sr)
