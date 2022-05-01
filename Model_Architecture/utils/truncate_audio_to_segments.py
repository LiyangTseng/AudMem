import os
import soundfile as sf
import librosa
from tqdm import tqdm

# hyperparameters
SR = 16000
# overlap time of new clips
HOP_TIME = 0.5
# new clip length
NEW_CLIP_LEN = 1 
ORIGINAL_DURATION = 5
AUDIO_ROOT = "data/raw_audios/"
OUTPUT_DIR = "data/1_second_clips"
augment_types = ["-1_semitones", "-2_semitones", "-3_semitones", "-4_semitones", "-5_semitones", "original",
"1_semitones", "2_semitones", "3_semitones", "4_semitones", "5_semitones"]

def truncate_audios():
    # truncate audios to multiple 1 second clips (with overlap)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for augment_type in tqdm(augment_types, desc="augment_types"):
        file_names = os.listdir(os.path.join(AUDIO_ROOT, augment_type))
        file_paths = [os.path.join(AUDIO_ROOT, augment_type, file_name) for file_name in file_names]
        for file_path in tqdm(file_paths, desc="files in {}".format(augment_type)):
            y, sr = librosa.load(file_path, sr=16000)
            start_time = 0
            cnt = 1
            while start_time+NEW_CLIP_LEN <= ORIGINAL_DURATION:
                end_time = start_time + NEW_CLIP_LEN
                audio_segments = y[int(start_time*SR) : int(end_time*SR)]
                output_path = os.path.join(OUTPUT_DIR, file_path.split("/")[-1].split(".")[0].split("intro_")[-1]+"_{}_{}.wav".format(augment_type, cnt))
                sf.write(output_path, audio_segments, SR)
                start_time += HOP_TIME
                cnt += 1



if __name__ == "__main__":
    truncate_audios()