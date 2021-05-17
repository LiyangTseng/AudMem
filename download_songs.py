import os
import json
output_dir = 'songs/'
with open('playlists.json') as json_file:
    playlists_dict = json.load(json_file)
for title  in playlists_dict:
    subdir=os.path.join(output_dir, title.replace(' ', '_'))
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    print(subdir)
    os.system('spotdl --output {output_subdir} {playlist_link} --ignore-ffmpeg-version '.format(
       output_subdir=subdir , playlist_link=playlists_dict[title]))
