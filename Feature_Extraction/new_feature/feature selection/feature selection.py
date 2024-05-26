from sklearn import feature_selection as fs
import pandas as pd
import numpy as np

df = pd.read_csv('./data_newFeature.csv')
col = [e for e in df.columns if e not in ('YT_id', 'segment_idx', 'augment_type','label')]

# https://stackoverflow.com/questions/39812885/retain-feature-names-after-scikit-feature-selection
def fs_VarianceThreshold(data, threshold=0.8*(1-0.8)):
    sel = fs.VarianceThreshold(threshold)
    sel.fit_transform(data)
    return data[data.columns[sel.get_support(indices=True)]]

def fs_regression(data,label,k):
    sel = fs.SelectKBest(fs.f_regression, k=k)
    sel.fit_transform(data,label)
    return data[data.columns[sel.get_support(indices=True)]]

# ans = fs_VarianceThreshold(df[col])
# a = ans.columns
k=20
ans2 = fs_regression(df[col],df['label'],k=k)
aa = ans2.columns
identify = ['YT_id', 'segment_idx', 'augment_type','label']
out = pd.merge(df[identify],ans2, left_index=True, right_index=True)
out['label'] = out.pop('label')
out.to_csv('./data_'+str(k)+'feature.csv', index=False)
print()