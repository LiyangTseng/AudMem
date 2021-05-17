import requests
import re

CLIENT_ID = '2f911a8c6d0947f4aa13ed53b8d8f44c'
CLIENT_SECRET = '4adb7a7db746484d98f216a8624cf7a9'

AUTH_URL = 'https://accounts.spotify.com/api/token'
# POST
auth_response = requests.post(AUTH_URL, {
    'grant_type': 'client_credentials',
    'client_id': CLIENT_ID,
    'client_secret': CLIENT_SECRET,
})

# convert the response to JSON
auth_response_data = auth_response.json()
# save the access token
access_token = auth_response_data['access_token']

headers = {
    'Authorization': 'Bearer {token}'.format(token=access_token)
}

# song_url = 'https://open.spotify.com/track/5piQeFWFRnI8S5tpcl58Hy?si=f6719ad094d640fa'
# split = re.split('[/?]', song_url)
# track_id = split[-2]
song_url = 'spotify:track:3BQHpFgAp4l80e1XslIjNI'
track_id = re.split(':', song_url)[-1]
# base URL of all Spotify API endpoints
BASE_URL = 'https://api.spotify.com/v1/'
# actual GET request with proper header
song = requests.get(BASE_URL + 'tracks/' + track_id, headers=headers)
song = song.json()

song_name = song['name']
song_album_id = song['album']['id']

print('song {} are from album with id {}'.format(song_name, song_album_id))
# spotify:album:5tjtkMg20jEuzuJrT27gBj
album = requests.get(BASE_URL + 'albums/' + song_album_id, headers=headers)
album = album.json()
print(album['genres'])