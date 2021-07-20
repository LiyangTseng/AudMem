import pandas as pd
import os

def open_all_links(idx):
    ''' a windows program to open all youtube links automatically '''

    file_path = 'C:\\Users\\User\\Downloads\\yt_info.csv'
    df = pd.read_csv(file_path)

    url_list = df['id'].values
    while idx < len(url_list):
        url = url_list[idx]
        print('url {} opened on browser, type something in cmd to open another url'.format(url))
        os.system('start "" {}'.format(url))    
        input()
        idx += 1

if __name__ == '__main__':
    start_idx = 405
    open_all_links(start_idx)