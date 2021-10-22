# usage: filter_vocals.sh input_dir output_dir

# for d in Audios/all_clips/*; do
for d in $1*; do
    spleeter separate -p spleeter:2stems -o $2 "$d" -f {filename}_{instrument}.{codec}
    audio_filename="$(cut -d'/' -f3 <<<"$d")"
    # only preserve accompaniment track
    # rm "Audios/all_clips_separated/${audio_filename:0:-4}/${audio_filename:0:-4}_vocals.wav"
    rm "$2/${audio_filename:0:-4}_vocals.wav"
    echo "remove vocals part from $audio_filename"
done

