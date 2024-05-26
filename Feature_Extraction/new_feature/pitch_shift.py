# ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8461686&tag=1
import os
import librosa
import pyrubberband
from tqdm import tqdm
import soundfile as sf

ORIGINAL_AUDIO_DIR = "./data/raw_audios/original"
SHIFT_SEMITONES = [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5]


def shift_pitch(y, semitone):
    return pyrubberband.pyrb.pitch_shift(y, sr, n_steps=semitone)


def tune_pitch(y, cents):
    pass


if __name__ == "__main__":

    original_raw_audios = os.listdir(ORIGINAL_AUDIO_DIR)

    for original_raw_audio in tqdm(original_raw_audios):
        original_audio_path = os.path.join(ORIGINAL_AUDIO_DIR, original_raw_audio)
        y, sr = librosa.load(original_audio_path)
        for semitone in SHIFT_SEMITONES:
            shift_audio_dir = "./data/raw_audios/{}_semitones".format(semitone)
            if not os.path.exists(shift_audio_dir):
                os.makedirs(shift_audio_dir)
            shifted_y = shift_pitch(y, semitone)
            shifted_audio_path = os.path.join(shift_audio_dir, original_raw_audio)
            sf.write(shifted_audio_path, shifted_y, sr)