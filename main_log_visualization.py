import glob
import os
import pandas as pd
import plotly.express as px
from AmoebaPlayGround.AmoebaTrainer import logs_folder

list_of_files = glob.glob(os.path.join(logs_folder, '*.log'))
latest_file = max(list_of_files, key=os.path.getctime)

df = pd.read_csv(latest_file, sep="\t")

fig = px.line(df, x='episode', y='loss', title='Loss')
fig.show()
