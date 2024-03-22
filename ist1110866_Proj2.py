import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from sklearn import metrics

#Load data
df_raw = pd.read_csv('NorthTower_data.csv')

df_forecast = df_raw.drop(columns=['Power_kWh'])
df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
df2 = df_forecast.iloc[:, 1:]
X2 = df2.values
fig1 = px.line(df_forecast, x="Date", y=df_forecast.columns[1:],
              color_discrete_map={
                   'Power-1': '#009de0',
                   'temp_C': 'orange',
                   'solarRad_W/m2': 'red',
                   'HDH': 'green'})

df_real = df_raw[['Date', 'Power_kWh']]
y2 = df_real['Power_kWh'].values

#Load models
with open('BT_model.pkl', 'rb') as file:
    BT_model = pickle.load(file)
y2_pred_BT = BT_model.predict(X2)

with open('RF_model.pkl', 'rb') as file:
    RF_model = pickle.load(file)
y2_pred_RF = RF_model.predict(X2)

with open('GB_model.pkl', 'rb') as file:
    GB_model = pickle.load(file)
y2_pred_GB = GB_model.predict(X2)

#Evaluate error metrics
MAE_BT = metrics.mean_absolute_error(y2, y2_pred_BT)
MBE_BT = np.mean(y2 - y2_pred_BT)
MSE_BT = metrics.mean_squared_error(y2, y2_pred_BT)
RMSE_BT = np.sqrt(MSE_BT)
cvRMSE_BT = RMSE_BT / np.mean(y2)
NMBE_BT = MBE_BT / np.mean(y2)

MAE_RF = metrics.mean_absolute_error(y2, y2_pred_RF)
MBE_RF = np.mean(y2 - y2_pred_RF)
MSE_RF = metrics.mean_squared_error(y2, y2_pred_RF)
RMSE_RF = np.sqrt(MSE_RF)
cvRMSE_RF = RMSE_RF / np.mean(y2)
NMBE_RF = MBE_RF / np.mean(y2)

MAE_GB = metrics.mean_absolute_error(y2, y2_pred_GB)
MBE_GB = np.mean(y2 - y2_pred_GB)
MSE_GB = metrics.mean_squared_error(y2, y2_pred_GB)
RMSE_GB = np.sqrt(MSE_GB)
cvRMSE_GB = RMSE_GB / np.mean(y2)
NMBE_GB = MBE_GB / np.mean(y2)

#print("BT Model Errors:", MAE_BT, MBE_BT, MSE_BT, RMSE_BT, cvRMSE_BT, NMBE_BT)
#print("RF Model Errors:", MAE_RF, MBE_RF, MSE_RF, RMSE_RF, cvRMSE_RF, NMBE_RF)
#print("GB Model Errors:", MAE_GB, MBE_GB, MSE_GB, RMSE_GB, cvRMSE_GB, NMBE_GB)

# Create a DataFrame with error metrics for all three models
df_metrics = pd.DataFrame({
    'Methods': ['Bootstrapping Regressor', 'Random Forest', 'Gradient Boosting'],
    'MAE': [MAE_BT, MAE_RF, MAE_GB],
    'MBE': [MBE_BT, MBE_RF, MBE_GB],
    'MSE': [MSE_BT, MSE_RF, MSE_GB],
    'RMSE': [RMSE_BT, RMSE_RF, RMSE_GB],
    'cvRMSE': [cvRMSE_BT, cvRMSE_RF, cvRMSE_GB],
    'NMBE': [NMBE_BT, NMBE_RF, NMBE_GB]})

# Create a DataFrame with prediction results
df_forecast = pd.DataFrame({
    'Date': df_real['Date'],
    'BT Regression': y2_pred_BT,
    'RF Regression': y2_pred_RF,
    'GB Regression': y2_pred_GB})

#print(df_metrics)
#print(df_forecast.head())

df_results = pd.merge(df_real, df_forecast, on='Date')
fig2 = px.line(df_results, 
               x='Date', 
               y=df_results.columns[1:], 
               title='Real vs Forecasted Results',
               color_discrete_map={
                   'BT Regression': '#009de0',
                   'RF Regression': 'orange',
                   'GB Regression': 'red'})


# Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)

app.layout = html.Div([
    html.H1('ist1110866 Forecast tool - Energy Services'),
    html.P('Representing Data, Forecasting, and Error metrics for January 01 until April 11, 2019.'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Error Metrics', value='tab-3'),]),
    html.Div(id='tabs-content')])

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        variable_selector = dcc.Dropdown(
            id='variable-selector',
            options=[{'label': col, 'value': col} for col in df_raw.columns if col != 'Date'],
            value='Power_kWh',  # Default to 'Power_kWh' or the first column
            multi=True,
            style={'width': '50%'})
        fig1.update_layout(
            xaxis_title="Date",
            yaxis_title="Values",)
        return html.Div([
            html.H3('Raw Data'),
            dcc.Graph(figure=fig1),
            variable_selector,
            html.Div(id='stats-output')])
    elif tab == 'tab-2':
        fig2.update_layout(
            xaxis_title="Date",
            yaxis_title="Values",)
        return html.Div([
            html.H3('Forecast'),
            dcc.Graph(figure=fig2)])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Error Metrics'),
            html.Label('Select Error Metrics:'),
            dcc.Dropdown(
                id='error-metrics-dropdown',
                options=[
                    {'label': col, 'value': col} for col in df_metrics.columns[1:]],
                value=['MAE'],
                multi=True,
                style={'width': '50%'}),
            html.Label('Select Regression Methods:'),
            dcc.Checklist(
                id='regression-methods-checkbox',
                options=[
                    {'label': method, 'value': method} for method in df_metrics['Methods']],
                value=[df_metrics['Methods'].iloc[0]],
                inline=True),
            html.Div(id='error-metrics-output')])

@app.callback(
    Output('stats-output', 'children'),
    [Input('variable-selector', 'value')])
def update_stats(selected_variables):
    if not selected_variables:
        return html.Div('Please select at least one variable.', style={'font-weight': 'bold'})
    if isinstance(selected_variables, str):
        selected_variables = [selected_variables]
    desired_order = ['Power_kWh', 'Power-1', 'temp_C', 'solarRad_W/m2', 'HDH']
    selected_variables = sorted(selected_variables, key=lambda x: desired_order.index(x) if x in desired_order else len(desired_order))
    try:
        stats = pd.concat([df_raw[var].describe().rename(var) for var in selected_variables], axis=1)
    except KeyError as e:
        return f"An error occurred: {e}"
    return generate_table(stats.reset_index())

@app.callback(
    Output('error-metrics-output', 'children'),
    [Input('error-metrics-dropdown', 'value'),
     Input('regression-methods-checkbox', 'value')])

def update_error_metrics_table(selected_metrics, selected_methods):
    if not selected_metrics or not selected_methods:
        return html.Div('Please select at least one error metric and one regression method.',style={'font-weight': 'bold'})
    filtered_metrics = df_metrics[df_metrics['Methods'].isin(selected_methods)]
    selected_metrics_df = filtered_metrics[['Methods'] + (selected_metrics if isinstance(selected_metrics, list) else [selected_metrics])]
    return generate_table(selected_metrics_df)

def generate_table(dataframe, max_rows=10):
    formatted_dataframe = dataframe.copy()
    for col in formatted_dataframe.select_dtypes(include=['float64']).columns:
        formatted_dataframe[col] = formatted_dataframe[col].apply(lambda x: f"{x:.12f}")
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in formatted_dataframe.columns])),
        html.Tbody([
            html.Tr([
                html.Td(formatted_dataframe.iloc[i][col]) for col in formatted_dataframe.columns
            ]) for i in range(min(len(formatted_dataframe), max_rows))])])

    
if __name__ == '__main__':
    app.run_server(debug=False, port=10886)