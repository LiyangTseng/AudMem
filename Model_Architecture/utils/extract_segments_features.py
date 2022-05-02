import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

full_audio_dir = "/media/lab812/53D8AD2D1917B29C/AudMem/dataset/AudMem/original_audios"
YT_ids = [file_name[:11] for file_name in os.listdir(full_audio_dir)]
segment_idx_list = [i+1 for i in range(9)]
augment_types = ["original", "-5_semitones", "-4_semitones", "-3_semitones", 
                            "-2_semitones", "-1_semitones", "1_semitones", "2_semitones", "3_semitones", 
                            "4_semitones", "5_semitones"]
audio_segments_dir = "data/1_second_clips"
chroma_dir = "data/1_second_features/chroma"
stretch_factor_path = "data/stretch_factors.csv"
SR = 16000


def extract_harmony_features(dict_data):
    """ extract haromonic and timbre features from segmented_clips """
    os.makedirs(chroma_dir, exist_ok=True)

    ids, segment_list, augment_list, chroma_means, chroma_stds = [], [], [], [], []
    pitch_names = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]
    C_means, Cs_means, D_means, Ds_means, E_means, F_means, Fs_means, G_means, Gs_means, A_means, As_means, B_means = [], [], [], [], [], [], [], [], [], [], [], []
    C_stds, Cs_stds, D_stds, Ds_stds, E_stds, F_stds, Fs_stds, G_stds, Gs_stds, A_stds, As_stds, B_stds = [], [], [], [], [], [], [], [], [], [], [], []
    

    for augment_type in tqdm(augment_types, desc="extracting harmony features"):
        for YT_id in tqdm(YT_ids, desc="augment type: " + augment_type, leave=False):
            for segment_idx in segment_idx_list:
                ids.append(YT_id)
                segment_list.append(segment_idx)
                augment_list.append(augment_type)

                audio_file_name = YT_id + "_" + augment_type + "_" + str(segment_idx) + ".wav"
                audio_file_path = os.path.join(audio_segments_dir, audio_file_name)
                y, sr = librosa.load(audio_file_path, sr=SR)
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                np.save(os.path.join(chroma_dir, YT_id + "_" + augment_type + "_" + str(segment_idx)), chroma)
                chroma_mean = np.mean(chroma, axis=1)
                chroma_std = np.std(chroma, axis=1)
                chroma_means.append(chroma_mean)
                chroma_stds.append(chroma_std)
                for i in range(len(pitch_names)):
                    exec("{}_means.append(chroma_mean[i])".format(pitch_names[i]))
                    exec("{}_stds.append(chroma_std[i])".format(pitch_names[i]))
            # break
        # break

    dict_data = {"YT_id": ids, "segment_idx": segment_list, "augment_type": augment_list,}
    for i in range(len(pitch_names)):
        pitch_name = pitch_names[i]
        dict_data[pitch_name + "_mean"] = eval(pitch_name + "_means")
     
    for i in range(len(pitch_names)):
        pitch_name = pitch_names[i]
        dict_data[pitch_name + "_std"] = eval(pitch_name + "_stds")
    
    return dict_data

def extract_tempo_features(dict_data):
    """ calculate bpm using full audio """
    assert dict_data != {}, "dict_data is should be filled in extract_harmony_features"
    # read stretch factor
    YT_id_to_streth_factor = {}
    with open(stretch_factor_path, "r") as f:
        next(f) # skip header
        for line in f:
            old_file_name, stretch_factor = line.strip().split(",")
            YT_id = old_file_name.split(".")[0][-11:]
            YT_id_to_streth_factor[YT_id] = float(stretch_factor)


    full_audio_files = os.listdir(full_audio_dir)
    YT_id_to_bpm = {}
    for YT_id in tqdm(YT_ids, desc="extracting tempo features"):
        full_audio_file = [file_path for file_path in full_audio_files if YT_id in file_path][0]                 
        y, sr = librosa.load(os.path.join(full_audio_dir, full_audio_file), sr=SR)
        bpm = librosa.beat.tempo(y=y, sr=sr)[0]*YT_id_to_streth_factor[YT_id]
        YT_id_to_bpm[YT_id] = bpm

    bpm_list = []
    for YT_id in tqdm(dict_data["YT_id"]):
        bpm_list.append(YT_id_to_bpm[YT_id])

    dict_data["bpm"] = bpm_list

    return dict_data

def extract_timbre_features(dict_data):
    """ extract timbre features from segmented_clips """
    spleeter_output_dir = "/media/lab812/53D8AD2D1917B29C/AudMem/dataset/sources_separated"
    source_components = ["vocals", "drums", "bass", "other"]
    vocals_intensities_mean, drums_intensities_mean, bass_intensities_mean, other_intensities_mean = [], [], [], []
    vocals_intensities_std, drums_intensities_std, bass_intensities_std, other_intensities_std = [], [], [], []
    # calculate energe of separated tracks 
    for augment_type in tqdm(augment_types, desc="extracting harmony features"):
        for YT_id in tqdm(YT_ids, desc="augment type: " + augment_type, leave=False):
            for segment_idx in segment_idx_list:
                audio_dir = YT_id + "_" + augment_type + "_" + str(segment_idx)
                for source in source_components:
                    audio_file_path = os.path.join(spleeter_output_dir, audio_dir, source+".wav")
                    if os.path.exists(audio_file_path):
                        y, sr = librosa.load(audio_file_path, sr=SR)
                        db = librosa.amplitude_to_db(y)
                        exec("{}_intensities_mean.append(db.mean())".format(source))
                        exec("{}_intensities_std.append(db.std())".format(source))
                    else:
                        exec("{}_intensities_mean.append(-1000)".format(source))
                        exec("{}_intensities_std.append(-1000)".format(source))


    dict_data["vocals_db_mean"] = vocals_intensities_mean
    dict_data["vocals_db_std"] = vocals_intensities_std
    dict_data["drums_db_mean"] = drums_intensities_mean
    dict_data["drums_db_std"] = drums_intensities_std
    dict_data["bass_db_mean"] = bass_intensities_mean
    dict_data["bass_db_std"] = bass_intensities_std
    dict_data["other_db_mean"] = other_intensities_mean
    dict_data["other_db_std"] = other_intensities_std

    return dict_data

def extract_emotion_features(dict_data):
    """ extract emotion features from segmented_clips """
    features_dir = "data/features"
    # assume that all emotions are the same after segmentation
    static_valences, static_arousals = [], []
    for augment_type in tqdm(augment_types, desc="extracting emotion features"):
        for YT_id in tqdm(YT_ids, desc="augment type: " + augment_type, leave=False):
            for segment_idx in segment_idx_list:
                static_valence = np.load(os.path.join(features_dir, augment_type, "emotions", "static_valence", "static_valence_normalize_5s_intro_" + YT_id + ".npy"))
                static_arousal = np.load(os.path.join(features_dir, augment_type, "emotions", "static_arousal", "static_arousal_normalize_5s_intro_" + YT_id + ".npy"))
                static_valences.append(static_valence)
                static_arousals.append(static_arousal)
    dict_data["static_valence"] = static_valences
    dict_data["static_arousal"] = static_arousals
    
    return dict_data

def get_labels(dict_data):
    """ get labels from csv file with labels """
    labels_file = "data/labels/track_memorability_scores_beta.csv"
    YT_id_to_label = {}
    with open(labels_file, "r") as f:
        next(f)
        for line in f:
            file_name, label = line.strip().split(",")
            YT_id = file_name.split(".")[0][-11:]
            YT_id_to_label[YT_id] = float(label)
    
    labels = []

    for augment_type in tqdm(augment_types, desc="copying labels"):
        for YT_id in tqdm(YT_ids, desc="augment type: " + augment_type, leave=False):
            for segment_idx in segment_idx_list:
                labels.append(YT_id_to_label[YT_id])

    dict_data["label"] = labels
    return dict_data

if __name__ == "__main__":
    
    # features_df = pd.read_csv("data/features.csv")
    # features_dict = features_df.to_dict(orient="list")
    
    features_dict = {}

    features_dict = extract_harmony_features(features_dict)
    features_dict = extract_tempo_features(features_dict)
    features_dict = extract_timbre_features(features_dict)
    features_dict = extract_emotion_features(features_dict)
    features_dict = get_labels(features_dict)

    features_df = pd.DataFrame(data=features_dict)
    print(features_df.head())
    features_df.to_csv("data/data.csv", index=False)

        