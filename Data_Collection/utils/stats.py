import argparse
import pandas as pd
import requests
from lxml.html import fromstring
import csv

def calculate_stats(args):
    ''' calculate stats from form result (transposed csv file) '''

    def get_filename(url):
        ''' 
        get audio file name from google drive link
        ref: https://stackoverflow.com/questions/26812470/how-to-get-page-title-in-requests 
        '''
        r = requests.get(url)
        tree = fromstring(r.content)
        return tree.findtext('.//title')[:-22]
    print('processing...')
    form_df = pd.read_csv(args.form)
    form_df = form_df[4:]
    sampled_yt_df = pd.read_csv(args.info)

    with open(args.output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['url', 'title', 'view', 'pop_score'])
        cnt = 1
        for idx, row in form_df.iterrows():
            url = list(row)[0].split(' ')[1]
            answers = list(row)[1:]

            # title = filename(url)[17:-4]
            score = answers.count('聽過')/len(answers)
            yt_id = "https://www.youtube.com/watch?v=" + get_filename(url)[5:16]


            title = sampled_yt_df[sampled_yt_df['id']==yt_id].iloc[0]['title']        
            view = sampled_yt_df[sampled_yt_df['id']==yt_id].iloc[0]['viewCount']        
            cnt += 1
            writer.writerow([yt_id, title, view, score])
    print('stats file stored at {}'.format(args.output))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--form', help='location of form csv file', default='AudMem_Form_3.csv')
    parser.add_argument('-i', '--info', help='location of yt_info file', default='../needed_yt_info.csv')
    parser.add_argument('-o', '--output', help='location of output stat file', default='stats_3.csv')
    args = parser.parse_args()

    calculate_stats(args)
