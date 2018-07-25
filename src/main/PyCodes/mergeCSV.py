import os
import pandas as pd
...
os.chdir('C:\\Users\\User\\Documents\\Sentiment Analysis')

dfs = [pd.read_csv(f, index_col=[0], parse_dates=[0])
        for f in os.listdir(os.getcwd()) if f.endswith('csv')]

finaldf = pd.concat(dfs, axis=0, join='inner').sort_index()
finaldf.to_csv("final.csv")