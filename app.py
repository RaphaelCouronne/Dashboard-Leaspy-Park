import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output


#%% Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.css.append_css({
    "external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"
})

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}




import plotly.express as px
import pandas as pd
import os
import sys
import plotly.graph_objects as go
import seaborn as sns
import sys
import numpy as np
from sklearn.decomposition import PCA
sys.path.append("../")
print(os.getcwd())

from leaspy import IndividualParameters, Data, Leaspy
from src.leaspype import get_reparametrized_ages,append_spaceshifts_to_individual_parameters_dataframe


#%% Load Data
path_datadashboard = os.path.join("app", "data")
leaspy = Leaspy.load(os.path.join(path_datadashboard, "leaspy.json"))
individual_parameters = IndividualParameters.load(os.path.join(path_datadashboard, "ip.csv"))
data = Data.from_csv_file(os.path.join(path_datadashboard, "data.csv"))
n_sources = leaspy.model.source_dimension
sources_name = ["sources_{}".format(i) for i in range(n_sources)]
df_data = data.to_dataframe().set_index(["ID","TIME"])
features = leaspy.model.features

# ind parameters
df_ind = individual_parameters.to_dataframe()
ind_param_names = list(df_ind.columns)+["pca1", "pca2"]
df_ind = df_ind.reset_index()

# PCA on the w_i
res = append_spaceshifts_to_individual_parameters_dataframe(df_ind, leaspy)[["w_{}".format(i) for i in range(leaspy.model.dimension)]]

pca = PCA(n_components=2)
pca.fit(res)
pca_res = pca.transform(res)
print(pca.explained_variance_ratio_)
df_ind["pca1"] = pca_res[:,0]
df_ind["pca2"] = pca_res[:,1]

# RBD+ / RBD-
rbd_cutoff = 6 / 13
is_RBD = df_data.groupby('ID').apply(lambda x: x['REM_total'].max() >= rbd_cutoff)
patients_rbd = list(is_RBD.index[is_RBD])
patients_rbd = [str(x) for x in patients_rbd]
patients_nonrbd = list(is_RBD.index[np.logical_not(is_RBD)])
patients_nonrbd = [str(x) for x in patients_nonrbd]
indices_groups = {'iPD-RBD+': patients_rbd, 'iPD-RBD-': patients_nonrbd}
df_ind["RBD"] = 1
idx_nonrbd = df_ind[np.isin(df_ind["ID"], indices_groups["iPD-RBD-"])].index
df_ind.loc[idx_nonrbd, "RBD"] = 0
#df_ind["RBD"] = df_ind["RBD"].astype('category')


ind_param_nice = ["Onset", "Speed",
                  "Source 1",
                  "Source 2",
                  "Source 3",
                  "Source 4",
                  "Source 5",
                  "PCA 1",
                  "PCA 2",]

ind_param_renaming = {
    'tau':"Onset",
    'xi':"Speed",
    'sources_0':"Source 1",
    'sources_1': "Source 2",
    'sources_2': "Source 3",
    'sources_3': "Source 4",
    'sources_4': "Source 5",
    "RBD": "RBD",
    "pca1": "PCA 1",
    "pca2": "PCA 2"
}

columns_mapping = {"MDS1_total": "MDS1",
                   "MDS2_total" : "MDS2",
                   "MDS3_off_total": "MDS3_off",
                   "SCOPA_total": "SCOPA",
                   "MOCA_total": "MOCA",
                   "REM_total": "REM",
                   "PUTAMEN_R": "Put. R",
                   "PUTAMEN_L": "Put. L",
                   "CAUDATE_R" : "Caud. R",
                   "CAUDATE_L" : "Caud. L"}

#%% Plot options
palette = sns.color_palette("colorblind", n_colors=11)
plotly_palette = {feature : f'rgb{tuple((np.array(palette[i])*255).astype(np.uint8))}' for i, feature in enumerate(features)}
plotly_palette_alpha = {feature : f'rgba{tuple((np.array(palette[i])*255).astype(np.uint8))}' for i, feature in enumerate(features)}
plotly_palette_02 = {key:val.split(")")[0]+",0.2)" for key,val in plotly_palette_alpha.items()}
plotly_palette_04 = {key:val.split(")")[0]+",0.4)" for key,val in plotly_palette_alpha.items()}

x_legend1 = 'Average trajectory : Physiological Age' +" "*10+'Patient trajectory : Age'
x_legend2 = 'Average trajectory : Physiological Age' +" "*10+'Patient trajectory : Physiological Age'

height_latent = 320
height_w = 250
range_w = 0.4

app.title = 'COVID 19 - World cases'
app.layout = html.H1("Yo my app !", className="header__text")

if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run(debug=True, port=int(os.environ.get("PORT", 5000)), host='0.0.0.0')
    #server = app.server
    #app.run(debug=True)


