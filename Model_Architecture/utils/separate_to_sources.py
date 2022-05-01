import os
from tqdm import tqdm

def source_separation(input_dir, output_dir):
    """ use spleeter to separate sources """
    full_audio_dir = "/media/lab812/53D8AD2D1917B29C/AudMem/dataset/AudMem/original_audios"
    YT_ids = [file_name[:11] for file_name in os.listdir(full_audio_dir)]
    augment_types = ["original", "-5_semitones", "-4_semitones", "-3_semitones", 
                            "-2_semitones", "-1_semitones", "1_semitones", "2_semitones", "3_semitones", 
                            "4_semitones", "5_semitones"]
    segment_idx_list = [i+1 for i in range(9)]

    os.makedirs(output_dir, exist_ok=True)
    for augment_type in tqdm(augment_types):
        for YT_id in YT_ids:
            for segment_idx in segment_idx_list:
                file_name = YT_id + "_" + augment_type + "_" + str(segment_idx) + ".wav"
                if file_name.endswith(".wav"):
                    # input()
                    if os.path.exists(os.path.join(output_dir, file_name.strip(".wav"))):
                        print("exists")
                        continue
                    else:
                        os.system("spleeter separate -o {} -p spleeter:4stems {}".format(os.path.join(output_dir), os.path.join(input_dir, file_name)))

if __name__ == "__main__":
    mix_input_dir = "data/1_second_clips"
    separated_sources_dir = "/media/lab812/53D8AD2D1917B29C/AudMem/dataset/sources_separated"
    source_separation(mix_input_dir, separated_sources_dir)
