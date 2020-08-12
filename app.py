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
path_datadashboard = "data/"
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

app.title = 'Dash Parkinson'

app.layout = html.Div([

        html.Div(
            html.H1("Leaspy - Parkinson's Disease"),
            style={"text-align": "center"}
        ),

        html.H2("Synthetic Parkinson Dataset"),


        html.Div(children=[

            html.Div(children=[

                html.Div([
                    html.Button('Reset', id='reset-indparam', n_clicks=0),
                    ], className="row"),
                #html.Button('Random Patient Data', id='random-patient-data', n_clicks=0),

                #html.Label('Show'),
                dcc.Checklist(
                    id="show_trj",
                    options=[
                        {'label': 'Average Trajectory', 'value': 'Avg-trj'},
                        {'label': 'Patient Trajectory', 'value': 'Pa-trj'},
                        {'label': 'Patient Data', 'value': 'Pa-data'},
                        {'label': 'Subgroup Trajectory', 'value': 'Sub-trj'},
                        {'label': 'Patient time reparametrization', 'value': 'Pa-reparam'},
                    ],
                    value=['Avg-trj']
                ),

                html.Label('Onset'),
                dcc.Slider(
                    id="Tau-Slider",
                    min=-20,
                    max=20,
                    # marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(3, 6)},
                    value=0,
                    updatemode="drag",
                    step=0.4
                ),

                html.Label('Speed'),
                dcc.Slider(
                    id="Xi-Slider",
                    min=-2,
                    max=2,
                    step=0.1,
                    # marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(3, 6)},
                    value=0,
                    updatemode="drag",
                ),

                html.Label('Source 1'),
                dcc.Slider(
                    id="source1-slider",
                    min=-3,
                    max=3,
                    step=0.1,
                    # marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(3, 6)},
                    value=0,
                    updatemode="drag",
                ),



                html.Label('Source 2'),
                dcc.Slider(
                    id="source2-slider",
                    min=-3,
                    max=3,
                    step=0.1,
                    # marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(3, 6)},
                    value=0,
                    updatemode="drag",
                ),

                html.Label('Source 3'),
                dcc.Slider(
                    id="source3-slider",
                    min=-3,
                    max=3,
                    step=0.1,
                    # marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(3, 6)},
                    value=0,
                    updatemode="drag",
                ),

                html.Label('Source 4'),
                dcc.Slider(
                    id="source4-slider",
                    min=-3,
                    max=3,
                    step=0.1,
                    # marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(3, 6)},
                    value=0,
                    updatemode="drag",
                ),

                html.Label('Source 5'),
                dcc.Slider(
                    id="source5-slider",
                    min=-3,
                    max=3,
                    step=0.1,
                    # marks={i: 'Label {}'.format(i) if i == 1 else str(i) for i in range(3, 6)},
                    value=0,
                    updatemode="drag",
                ),


            ],
                className="two columns"
            ),


            html.Div(children=[

                dcc.Graph(
                    id='patient-trajectory',
                ),

                dcc.Input(id='x-legend', value=x_legend1, type='text', style={'width': 600}),

                dcc.RangeSlider(
                    id='age-range-slider',
                    min=20,
                    max=100,
                    step=0.5,
                    value=[50, 80]
                ),

                html.Label('Select Clinical Scores'),
                dcc.Dropdown(
                    id="features-select",
                    options=[{'label': columns_mapping[feature], 'value': feature} for feature in features],
                    value=features,
                    multi=True
                ),

                html.Label('Select Patients'),
                dcc.Dropdown(
                    id="patients-select",
                    options=[{"label": x, "value": x} for x in list(df_data.index.unique("ID"))],
                    value=["3001"],
                    multi=True,
                    style={'height': '30px'}
                ),

            ],
                className="seven columns"
            ),



            html.Div(children=[


                #dcc.Graph(
                #    id='xitau-latent-space',
                #),

                dcc.Graph(
                    id='sources-latent-space',
                ),

                html.Div([

                dcc.Dropdown(
                    id="abs-select",
                    options=[{"label": x_nice, "value": x} for x, x_nice in zip(ind_param_names, ind_param_nice)],
                    value="tau",
                    multi=False),

                dcc.Dropdown(
                    id="ord-select",
                    options=[{"label": x_nice, "value": x} for x, x_nice in zip(ind_param_names, ind_param_nice)],
                    value="xi",
                    multi=False),

                dcc.Dropdown(
                    id="color-select",
                    options=[{"label": "No color", "value": "no color"}]+
                            [{"label": "RBD", "value": "RBD"}]+
                            [{"label": x_nice, "value": x} for x, x_nice in zip(ind_param_names, ind_param_nice)]
                            ,
                    value="no color",
                    multi=False),

                    html.Div([
                        html.Label("Subgroup Sources"),
                        dcc.Graph(
                            id='sources-heatmap',
                        ),

                    ]),


                    ]),


            ],
                 className="three columns"
            ),



        ], className="row"),


        html.Div(children=[



            #html.Div([
            #    dcc.Input(id='patients', value='', type='text')
            #],
            #         className="two columns"),



            html.Div([


            ],
                className="seven columns"
            ),



            ],
        className="row"),


        #html.Div("(Add PCA coefs, add Reparametrized Time, add drag and drop of data,indparam,leaspy)")

        #html.H2("2. Replication Analysis : PPMI/ICEBERG/DIGPD"),

        ],
    )



@app.callback(
    Output('Tau-Slider', 'value'),
    [Input('reset-indparam', 'n_clicks')])
def reset_source(reset_clicks):
    return 0


@app.callback(
    Output('Xi-Slider', 'value'),
    [Input('reset-indparam', 'n_clicks')])
def reset_source(n_clicks):
    return 0

@app.callback(
    Output('source1-slider', 'value'),
    [Input('reset-indparam', 'n_clicks')])
def reset_source(n_clicks):
    return 0

@app.callback(
    Output('source2-slider', 'value'),
    [Input('reset-indparam', 'n_clicks')])
def reset_source(n_clicks):
    return 0

@app.callback(
    Output('source3-slider', 'value'),
    [Input('reset-indparam', 'n_clicks')])
def reset_source(n_clicks):
    return 0

@app.callback(
    Output('source4-slider', 'value'),
    [Input('reset-indparam', 'n_clicks')])
def reset_source(n_clicks):
    return 0

@app.callback(
    Output('source5-slider', 'value'),
    [Input('reset-indparam', 'n_clicks')])
def reset_source(n_clicks):
    return 0

@app.callback(
    Output('age-range-slider', 'value'),
    [Input('reset-indparam', 'n_clicks')])
def reset_source(n_clicks):
    return [40,90]

@app.callback(
    Output('x-legend', 'value'),
    [Input('show_trj', 'value')])
def change_legend(show_trj):
    if "Pa-reparam" not in show_trj:
        return x_legend1
    else:
        return x_legend2


@app.callback(
    Output('sources-heatmap', 'figure'),
    [Input('patients-select', 'value')])
def update_heatmap(patients_select):
    features_nice = [columns_mapping[x] for x in features]
    if len(patients_select)>0:
        df_ind_sub = pd.DataFrame(append_spaceshifts_to_individual_parameters_dataframe(pd.DataFrame(df_ind[np.isin(df_ind["ID"],patients_select)]), leaspy)[
            ["w_{}".format(i) for i in range(leaspy.model.dimension)]].mean())
        df_ind_sub.index = features_nice
    else:
        df_ind_sub = pd.DataFrame([0]*leaspy.model.dimension, index=features_nice)
    fig = px.imshow(-df_ind_sub.T,
                    color_continuous_scale="RdYlGn", range_color=[-range_w, range_w],
                    height=height_w)

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="LightSteelBlue",
        autosize=True
    )
    return fig

@app.callback(
    Output('patient-trajectory', 'figure'),
    [Input('features-select', 'value'),
     Input('show_trj', 'value'),
     Input('patients-select', 'value'),
     Input('Tau-Slider', 'value'),
     Input('Xi-Slider', 'value'),
     Input('source1-slider', 'value'),
     Input('source2-slider', 'value'),
     Input('source3-slider', 'value'),
     Input('source4-slider', 'value'),
     Input('source5-slider', 'value'),
     Input('age-range-slider', "value"),
     ])
def plot_main_fig(features_plot, show_trj,
               patients_selected,
               Tau_value,
               Xi_value,
               Source1,
               Source2,
               Source3,
               Source4,
               Source5,
               age_range,
               ):

    # Select patient
    if len(patients_selected)>0:
        patient_IDs = patients_selected
    else:
        patient_IDs = []

    fig2 = go.Figure()#layout=go.Layout(height=600, width=800))
    fig2.update_yaxes(range=[-0.05, 1.05])
    fig2.update_xaxes(range=[age_range[0], age_range[1]])
    timepoints = {'mean': np.linspace(age_range[0], age_range[1], 20)}
    show_legend = True

    if 'Avg-trj' in show_trj:



        ip_dict_modified = {
            'xi': individual_parameters.get_mean('xi'),
            'tau': individual_parameters.get_mean('tau'),
            'sources': individual_parameters.get_mean('sources')
        }

        sources = [Source1, Source2, Source3, Source4, Source5]

        ip_dict_modified["tau"] = ip_dict_modified["tau"] + Tau_value
        ip_dict_modified["xi"] = ip_dict_modified["xi"] + Xi_value
        for i in range(5):
            ip_dict_modified["sources"][i] = ip_dict_modified["sources"][i] + sources[i]

        ip_modified = IndividualParameters()
        ip_modified.add_individual_parameters('mean', ip_dict_modified)
        df_avg_modified = pd.DataFrame(leaspy.estimate(timepoints, ip_modified)["mean"], index=timepoints["mean"], columns=list(columns_mapping.keys())).reset_index()
        df_avg_modified.rename(columns={"index": "TIME"}, inplace=True)


    # Average Trajectory
        for i, feature in enumerate(features_plot):
            fig2.add_trace(go.Scatter(x=df_avg_modified["TIME"], y=df_avg_modified[feature],
                                      line=dict(color=plotly_palette_04[feature], width=6),
                                      name=columns_mapping[feature],
                                      showlegend=show_legend
                                      ))
        show_legend = False

    if 'Sub-trj' in show_trj and len(patient_IDs)>0:
        ip_sub = individual_parameters.subset(patient_IDs)
        ip_sub_mean_dict = {
            'xi': ip_sub.get_mean('xi'),
            'tau': ip_sub.get_mean('tau'),
            'sources': ip_sub.get_mean('sources')
        }

        ip_sub_mean = IndividualParameters()
        ip_sub_mean.add_individual_parameters('mean', ip_sub_mean_dict)
        df_avg_sub = pd.DataFrame(leaspy.estimate(timepoints, ip_sub_mean)["mean"], index=timepoints["mean"], columns=list(columns_mapping.keys())).reset_index()
        df_avg_sub.rename(columns={"index": "TIME"}, inplace=True)

        for i, feature in enumerate(features_plot):
            fig2.add_trace(go.Scatter(x=df_avg_sub["TIME"], y=df_avg_sub[feature],
                                      line=dict(color=plotly_palette_04[feature], width=6),
                                      name=columns_mapping[feature],
                                      showlegend=show_legend
                                      ))
        show_legend = False


    if 'Pa-trj' in show_trj:
    # Individual Trajectory
        for patient_ID in patient_IDs:
            pa_time = data.get_by_idx(patient_ID).timepoints

            duration = max(pa_time) - min(pa_time)
            pa_timepoints = np.linspace(min(pa_time) - duration / 2, max(pa_time) + duration / 2, 20)


            ip = individual_parameters[patient_ID]
            patient_traj = leaspy.estimate({patient_ID: pa_timepoints}, {patient_ID: ip})[patient_ID]

            if 'Pa-reparam' in show_trj:
                pa_timepoints = get_reparametrized_ages({patient_ID: pa_timepoints},
                                                                   individual_parameters,
                                                                   leaspy)[patient_ID]

            df_patient = pd.DataFrame(patient_traj, index=pa_timepoints, columns=features).reset_index()
            df_patient.rename(columns={"index": "TIME"}, inplace=True)

            for i, feature in enumerate(features_plot):
                fig2.add_trace(go.Scatter(x=df_patient["TIME"], y=df_patient[feature],
                                          mode='lines',
                                          name=columns_mapping[feature],
                                          line=dict(color=plotly_palette[feature], width=2),
                                          showlegend=show_legend))
            show_legend = False

    if "Pa-data" in show_trj:
        for patient_ID in patient_IDs:
            # Patient Data
            df_patientdata = df_data.loc[patient_ID].reset_index()

            pa_timepoints = data.individuals[patient_ID].timepoints

            if 'Pa-reparam' in show_trj:
                pa_timepoints = get_reparametrized_ages({patient_ID: pa_timepoints},
                                                                   individual_parameters,
                                                                   leaspy)[patient_ID]



            for i, feature in enumerate(features_plot):
                fig2.add_trace(go.Scatter(x=pa_timepoints, y=df_patientdata[feature],
                                          mode='lines+markers',
                                          line=dict(color=plotly_palette[feature], width=1),
                                          marker=dict(size=8),
                                          name=columns_mapping[feature],
                                          showlegend=show_legend,
                                          connectgaps=True))
            show_legend = False


    fig2.update_layout(
        clickmode='event+select',
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="LightSteelBlue",
        autosize=True
    )

    return fig2




def plot_latent(xlabel, ylabel, color):

    if color =="no color":
        fig = px.scatter(df_ind.rename(columns=ind_param_renaming),
                         x=ind_param_renaming[xlabel],
                         y=ind_param_renaming[ylabel], height=height_latent, custom_data=["ID"])

        fig.update_traces(marker={'color': 'rgba(0, 116, 217, 0.7)', 'size': 6})

    else:
        if color == "RBD":
            fig = px.scatter(df_ind.rename(columns=ind_param_renaming),
                             x=ind_param_renaming[xlabel],
                             y=ind_param_renaming[ylabel],
                             color=ind_param_renaming[color],
                             height=height_latent, custom_data=["ID"],
                             color_continuous_scale=["Blue", "Red"])
                             #color_discrete_sequence=["Red", "Blue"])
        else:
            fig = px.scatter(df_ind.rename(columns=ind_param_renaming),
                             x=ind_param_renaming[xlabel],
                             y=ind_param_renaming[ylabel],
                             color=ind_param_renaming[color],
                             height=height_latent, custom_data=["ID"],
                             color_continuous_scale='Bluered_r')


        fig.update_traces(marker={ 'size': 6})

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="LightSteelBlue",
        autosize=True,
        clickmode='event+select'
    )

    return fig

@app.callback(
    Output('sources-latent-space', 'figure'),
    [Input('patients-select', 'value'),
     Input('abs-select', 'value'),
     Input('ord-select', 'value'),
     Input('color-select', 'value'),
     Input('Tau-Slider', 'value'),
     Input('Xi-Slider', 'value'),
     Input('source1-slider', 'value'),
     Input('source2-slider', 'value'),
     Input('source3-slider', 'value'),
     Input('source4-slider', 'value'),
     Input('source5-slider', 'value')
     ]
)
def update_source_selection(patients_selection, abs, ord, color, Tau_value,
               Xi_value,
               Source1,
               Source2,
               Source3,
               Source4,
               Source5,
                            ):

    fig = plot_latent(abs, ord, color)
    df_avg_ind = df_ind.mean()
    df_avg_ind["tau"] += Tau_value
    df_avg_ind["xi"] += Xi_value

    sources = [Source1, Source2, Source3, Source4, Source5]
    for i in range(n_sources):
        df_avg_ind["sources_{}".format(i)] += sources[i]

    # Update PCA
    w = leaspy.model.attributes.mixing_matrix.numpy().dot(np.array(sources))
    pca_res = pca.transform(w.reshape(1,-1))
    print(pca_res)
    df_avg_ind["pca1"] += float(pca_res[0][0])
    df_avg_ind["pca2"] += float(pca_res[0][1])


    if patients_selection is not None:
        #election_xitau = [p['pointIndex'] for p in selection_xitau['points']]

        patients_index = df_ind[np.isin(df_ind["ID"], patients_selection)].index
        fig.update_traces(selectedpoints=patients_index,
                          mode='markers+text',
                          #marker={'color': 'rgba(0, 116, 217, 0.7)', 'size': 10},
                          marker={'size': 10},
                          unselected={'marker': {'opacity': 0.2},
                                      'textfont': {'color': 'rgba(0, 0, 0, 0)' }})

        df_avg_sub_ind = df_ind.set_index("ID").loc[patients_selection].mean()
        fig.add_trace(go.Scatter(x=[df_avg_sub_ind[abs]], y=[df_avg_sub_ind[ord]],
                                 mode='markers',
                                 line=dict(color="Black", width=8),
                                 marker=dict(size=16, opacity=0.9),
                                 showlegend=False))


    fig.add_trace(go.Scatter(x=[df_avg_ind[abs]], y=[df_avg_ind[ord]],
                              mode='markers',
                              line=dict(color="purple", width=10),
                              marker=dict(size=16, opacity=0.6),
                              showlegend=False))

    return fig


@app.callback(
    Output('patients-select', 'value'),
    [Input('sources-latent-space', 'selectedData')]
)


def update_patientselect_selection(selection_sources):


    patient_sources = []

    if selection_sources is not None:
        patient_sources = np.array([p['customdata'] for p in selection_sources['points'] if 'customdata' in p.keys()]).reshape(-1).tolist()

    return patient_sources


if __name__ == '__main__':
    app.run_server(debug=True)
    #app.run(debug=True, port=int(os.environ.get("PORT", 5000)), host='0.0.0.0')
    #server = app.server
    #app.run(debug=True)


