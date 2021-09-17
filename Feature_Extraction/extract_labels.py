import os
import pandas as pd
import csv
# TODO: value depend on pilot study
VIGILANCE_THRESHOLD = 0.1
TRACK_NUM = 94
# idx of audio
VIGILANCE_AUDIO_IDX = [1, 12, 24, 39, 47, 61, 74, 80, 84, 91]
# idx of audio
TARGET_AUDIO_IDX = [0, 3, 4, 5, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 26, 27, 28, 29, 30, 32,
 34, 37, 38, 40, 43, 44, 45, 48, 50, 51, 52, 53, 55, 56, 57, 58, 60, 62, 64, 65, 66, 67, 70, 72, 73, 75, 76, 77, 78, 81, 82, 83, 86, 89]

SLOT_ORDER = [0, 1, 2, 1, 3, 4, 0, 5, 6, 7, 3, 8, 9, 10, 5, 11, 12, 7, 13, 12, 14, 10, 15, 16, 
  17, 18, 19, 20, 21, 22, 23, 15, 24, 25, 24, 18, 26, 17, 27, 28, 29, 23, 30, 31, 27, 32, 33, 34, 29, 
  35, 36, 32, 37, 38, 21, 39, 40, 9, 39, 41, 42, 40, 43, 44, 20, 34, 45, 26, 46, 47, 43, 48, 49, 47, 
  50, 51, 14, 48, 52, 45, 53, 50, 54, 55, 56, 57, 51, 58, 59, 55, 28, 60, 61, 62, 61, 63, 58, 64, 62, 
  65, 66, 60, 67, 68, 65, 69, 70, 66, 4, 71, 72, 44, 11, 73, 52, 19, 74, 75, 72, 74, 76, 70, 38, 73, 77, 
  67, 64, 56, 78, 13, 76, 75, 37, 79, 80, 77, 80, 78, 30, 81, 57, 53, 82, 83, 84, 85, 86, 84, 87, 22, 82,
   81, 88, 83, 89, 90, 91, 86, 91, 92, 93, 89]

def get_track_index(index_file):
    ''' return a dictionary of mapping between index and audio filename'''
    with open(index_file, 'r') as f:
        lines = f.readlines()
        id2track_dict = {int(line.strip().split()[0]): line.strip().split()[1] for line in lines}
        return id2track_dict

def preprocess_filtering(experiment_df):
    ''' return qualified experiment data that filter repeated participations, incomplete fields '''
    # drop rows with NaN column
    experiment_df = experiment_df.dropna()
    # only keep the very first userEmail
    experiment_df = experiment_df.drop_duplicates(subset='userEmail', keep='first')

    return experiment_df

def check_track_memorized(singleUserData, trackMemorability_dict, vigilanceThreshold=VIGILANCE_THRESHOLD):
    ''' check if targets are memorized, return track_memorability_dict to update '''
    # convert to list of int, each element representing track index
    audioOrder = list(map(int, (singleUserData.loc['audioOrder']).split(','))) 
    userResponse = list(map(int, (singleUserData.loc['userResponse']).split(','))) 

    TARGET_PAIRS = []
    VAGILANCE_PAIRS = []

    ''' ==== filter unqualified user ===='''
    
    for idx in range(TRACK_NUM):
        if idx in VIGILANCE_AUDIO_IDX:
            # ref: https://stackoverflow.com/questions/9542738/python-find-in-list
            occureance_slots = [slot_idx for slot_idx,track_idx in enumerate(SLOT_ORDER) if track_idx==idx]
            VAGILANCE_PAIRS.append(occureance_slots)

    vigilanceMemorized_list = []
    for vigilance_pair in VAGILANCE_PAIRS:
        memorized = 1 if (userResponse[vigilance_pair[0]]==0 and userResponse[vigilance_pair[1]]==1) else 0
        vigilanceMemorized_list.append(memorized)

    vigilancePassed = float(vigilanceMemorized_list.count(1))/len(vigilanceMemorized_list) > vigilanceThreshold
    # print(vigilance_memorized_list, vigilancePassed)
    if not vigilancePassed:
        return trackMemorability_dict
        ''' ================================='''
    else:
        for idx in range(TRACK_NUM):
            if idx in TARGET_AUDIO_IDX:
                occureance_slots = [slot_idx for slot_idx,track_idx in enumerate(SLOT_ORDER) if track_idx==idx]
                TARGET_PAIRS.append(occureance_slots)

        for vigilance_pair in TARGET_PAIRS:
            track_idx = audioOrder[SLOT_ORDER[vigilance_pair[0]]]
            track_name = id_to_track[track_idx]
            repeat_length = vigilance_pair[1]-vigilance_pair[0]
            memorized = 1 if (userResponse[vigilance_pair[0]]==0 and userResponse[vigilance_pair[1]]==1) else 0
            trackMemorability_dict[track_name].append({repeat_length: memorized})
        
        return trackMemorability_dict

def write_labels(track_memorability_dict):
    ''' write lables to output directory '''
    label_file = os.path.join('labels', 'track_memorability.csv')
    with open(label_file, 'w') as csv_file:  
        writer = csv.writer(csv_file)
        writer.writerow(['track', 'repeat_interval:memorized'])
        for track, memorability in track_memorability_dict.items():
            writer.writerow([track, memorability])
    print('labels saved at {}'.format(label_file))

if __name__ == '__main__':

    id_to_track = get_track_index('track_index.txt')
    trackMemorability_dict = {track: [] for idx, track in id_to_track.items()}
    experimentData = pd.read_csv('experimentData_beta.csv')
    qualifiedExperimentData = preprocess_filtering(experimentData)
    
    for index, singleUserData in qualifiedExperimentData.iterrows():
        trackMemorability_dict = check_track_memorized(singleUserData, trackMemorability_dict)
    
    write_labels(trackMemorability_dict)

    