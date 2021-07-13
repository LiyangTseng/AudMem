import requests
import random
import string
import pandas as pd
from tqdm import tqdm
import sys

MAX_RESULTS = 50
# YOUTUBE_API_KEY ='AIzaSyBkUAaWkEqEmuBH_ROYkiqmTO-JlG2WaH8'
YOUTUBE_API_KEY ='AIzaSyC57lKSM8wFI-tcy4__ZJSx6F6DHa8q8i4'
BASE_URL = "https://www.googleapis.com/youtube/v3/"

def get_statistics(vid_df):
    ''' get statistics for videos and append in the dataframe'''
    print('Generating statistics')
    viewCount_list =  []
    for _, row in tqdm(vid_df.iterrows(), total=vid_df.shape[0]):
        vid_id = row['id']
        path = 'videos?part={}&id={}'.format('snippet,statistics,contentDetails,topicDetails',vid_id)
        api_url = '{base_url}{path}&key={api_key}'.format(base_url=BASE_URL, path=path, api_key=YOUTUBE_API_KEY)
        r = requests.get(api_url)
        if r.status_code == requests.codes.ok:
            data = r.json()
        else:
            raise Exception('Requests Error')
            data = None
        statistics_dict = data['items'][0]['statistics']
        viewCount_list.append(statistics_dict['viewCount'])
        
    vid_df['viewCount'] = viewCount_list

    return vid_df

def get_video_categorie_id(category):
    ''' video categories '''
    path = 'videoCategories?part={part}&regionCode={regionCode}&maxResults={maxResults}'.format(
        part='snippet', regionCode='US', maxResults=MAX_RESULTS)
    api_url = '{base_url}{path}&key={api_key}'.format(base_url=BASE_URL, path=path, api_key=YOUTUBE_API_KEY)
    r = requests.get(api_url)
    if r.status_code == requests.codes.ok:
        data = r.json()
    else:
        raise Exception('Requests Error')
    category_dict = {}
    for item in data['items']:
        category_dict[item['id']] = item['snippet']['title']
    category_id = list(category_dict.keys())[list(category_dict.values()).index(category)]
    return category_id


def get_random_videos(category_id):
    ''' get random video by random query '''
    random_q = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(3))
    path = "search?key={}&maxResults={}&part=snippet&type=video&q={}&videoCategoryId={}".format(
        YOUTUBE_API_KEY,MAX_RESULTS,random_q,category_id)

    api_url = '{base_url}{path}&key={api_key}'.format(base_url=BASE_URL, path=path, api_key=YOUTUBE_API_KEY)
    r = requests.get(api_url)
    if r.status_code == requests.codes.ok:
        data = r.json()
    else:
        raise Exception('Requests Error')

    vid_info_df = pd.DataFrame(columns=['id', 'title'])
    i = 0
    for item in data['items']:
        video_id = item['id']['videoId']
        title = item['snippet']['title']
        vid_info_df.loc[i] = [video_id] + [title]
        i += 1
    print('{} random videos collected'.format(MAX_RESULTS))
    return vid_info_df

def collect_yt_data(video_num):
    
    print('Collecting {} videos'.format(video_num))
    save_file = 'yt_clips.csv'
    ''' 
        get random music-category youtube videos from api and record their statistics, 
        the results will be saved at [save_file]    
    '''

    music_category_id = get_video_categorie_id('Music')
    total_video_info_df = pd.DataFrame(columns=['id', 'title'])
    for i in range(video_num//MAX_RESULTS):
        video_info_df = get_random_videos(music_category_id)
        total_video_info_df = total_video_info_df.append(video_info_df, ignore_index=True)
    total_video_info_df = get_statistics(total_video_info_df)
    total_video_info_df['id'] = total_video_info_df['id'].apply(lambda vid_id: 'https://www.youtube.com/watch?v='+vid_id)
    total_video_info_df.to_csv(save_file, index=False)
    print('Saving data to {}'.format(save_file))

if __name__ == '__main__':
    collect_yt_data(int(sys.argv[1]))    
