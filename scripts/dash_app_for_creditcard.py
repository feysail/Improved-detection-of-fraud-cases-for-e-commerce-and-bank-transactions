from flask import Flask, jsonify
import pandas as pd
from dash import Dash, dcc, html
import plotly.express as px

server = Flask(__name__)


data = pd.read_csv(r'C:\Users\ASUS VIVO\Desktop\e-commerce\Improved-detection-of-fraud-cases-for-e-commerce-and-bank-transactions\data\data\Fraud_data_1.csv')

# Flask endpoint for summary statistics
@server.route('/api/summary', methods=['GET'])
def summary():
    total_transactions = data.shape[0]
    total_frauds = data[data['is_fraud'] == 1].shape[0]
    fraud_percentage = (total_frauds / total_transactions) * 100

    return jsonify({
        'total_transactions': total_transactions,
        'total_frauds': total_frauds,
        'fraud_percentage': fraud_percentage
    })

# Create a Dash app
app = Dash(__name__, server=server, url_base_pathname='/dashboard/')

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Fraud Detection Dashboard"),
    html.Div(id='summary-boxes'),
    dcc.Graph(id='fraud-trend-chart'),
    dcc.Graph(id='fraud-device-chart'),
    dcc.Graph(id='fraud-browser-chart'),
])


@app.callback(
    dash.dependencies.Output('summary-boxes', 'children'),
    [dash.dependencies.Input('summary-boxes', 'id')]
)
def update_summary_boxes(_):
    summary_data = summary().get_json()
    return html.Div([
        html.Div(f"Total Transactions: {summary_data['total_transactions']}"),
        html.Div(f"Total Fraud Cases: {summary_data['total_frauds']}"),
        html.Div(f"Fraud Percentage: {summary_data['fraud_percentage']:.2f}%")
    ])


@app.callback(
    dash.dependencies.Output('fraud-trend-chart', 'figure'),
    [dash.dependencies.Input('fraud-trend-chart', 'id')]
)
def update_fraud_trend_chart(_):
    fraud_trend = data.groupby('date')['is_fraud'].sum().reset_index()
    fig = px.line(fraud_trend, x='date', y='is_fraud', title='Fraud Cases Over Time')
    return fig


@app.callback(
    dash.dependencies.Output('fraud-device-chart', 'figure'),
    [dash.dependencies.Input('fraud-device-chart', 'id')]
)
def update_fraud_device_chart(_):
    device_counts = data[data['is_fraud'] == 1].groupby('device')['is_fraud'].count().reset_index()
    fig = px.bar(device_counts, x='device', y='is_fraud', title='Fraud Cases by Device')
    return fig


@app.callback(
    dash.dependencies.Output('fraud-browser-chart', 'figure'),
    [dash.dependencies.Input('fraud-browser-chart', 'id')]
)
def update_fraud_browser_chart(_):
    browser_counts = data[data['is_fraud'] == 1].groupby('browser')['is_fraud'].count().reset_index()
    fig = px.bar(browser_counts, x='browser', y='is_fraud', title='Fraud Cases by Browser')
    return fig

if __name__ == '__main__':
    app.run(debug=True)