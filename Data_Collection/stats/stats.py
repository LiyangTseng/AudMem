import pandas as pd
import requests
from lxml.html import fromstring
import csv

def filename(url):
    ''' https://stackoverflow.com/questions/26812470/how-to-get-page-title-in-requests '''
    r = requests.get(url)
    tree = fromstring(r.content)
    return tree.findtext('.//title')[:-22]

form_df = pd.read_csv('AudMem_Form.csv')
form_df = form_df[3:]
sampled_yt_df = pd.read_csv('../sampled_yt_info.csv')

with open('stats.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['url', 'title', 'mem_score', 'view'])
    cnt = 1
    for idx, row in form_df.iterrows():
        url = list(row)[0].split(' ')[1]
        answers = list(row)[1:]

        # title = filename(url)[17:-4]
        score = answers.count('聽過')/len(answers)
        yt_id = "https://www.youtube.com/watch?v=" + filename(url)[5:16]


        title = sampled_yt_df[sampled_yt_df['id']==yt_id].iloc[0]['title']        
        view = sampled_yt_df[sampled_yt_df['id']==yt_id].iloc[0]['viewCount']        
        print(cnt, yt_id, title, view, score)
        cnt += 1
        writer.writerow([yt_id, title, view, score])

