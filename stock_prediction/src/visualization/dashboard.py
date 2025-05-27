import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

class TradingDashboard:
    """Dashboard interactif Dash pour visualisation en temps réel"""
    
    def __init__(self, config):
        self.config = config
        self.app = dash.Dash(__name__)
        self.logger = logging.getLogger(__name__)
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Configure le layout du dashboard"""
        
        self.app.layout = html.Div([
            html.Div([
                html.H1("Tableau de Bord de Trading Intelligent", className="header-title"),
                html.Div(id="last-update", className="header-subtitle")
            ], className="header"),
            
            html.Div([
                # Sélection des symboles
                html.Div([
                    html.Label("Sélectionner un symbole:"),
                    dcc.Dropdown(
                        id="symbol-dropdown",
                        options=[{"label": s, "value": s} for s in self.config['data']['symbols']],
                        value=self.config['data']['symbols'][0],
                        className="dropdown"
                    )
                ], className="control-item"),
                
                # Sélection de la période
                html.Div([
                    html.Label("Période:"),
                    dcc.DatePickerRange(
                        id="date-range",
                        start_date=datetime.now() - timedelta(days=365),
                        end_date=datetime.now(),
                        display_format='YYYY-MM-DD',
                        className="date-picker"
                    )
                ], className="control-item"),
                
                # Boutons de rafraîchissement
                html.Button("Rafraîchir", id="refresh-button", className="button"),
                
            ], className="controls"),
            
            # Graphiques principaux
            html.Div([
                # Graphique des prix avec indicateurs
                dcc.Graph(id="price-chart", className="main-chart"),
                
                # Indicateurs techniques
                html.Div([
                    dcc.Graph(id="rsi-chart", className="indicator-chart"),
                    dcc.Graph(id="macd-chart", className="indicator-chart"),
                ], className="indicator-row"),
                
                # Volume
                dcc.Graph(id="volume-chart", className="volume-chart"),
                
            ], className="charts-container"),
            
            # Tableau des signaux et métriques
            html.Div([
                html.Div(id="signals-table", className="signals-container"),
                html.Div(id="metrics-table", className="metrics-container"),
            ], className="tables-container"),
            
            # Prévisions
            html.Div([
                html.H3("Prévisions Multi-Modèles"),
                dcc.Graph(id="forecast-chart", className="forecast-chart"),
                html.Div(id="forecast-confidence", className="forecast-info")
            ], className="forecast-container"),
            
            # Performance du backtest
            html.Div([
                html.H3("Performance Historique"),
                dcc.Graph(id="backtest-chart", className="backtest-chart"),
                html.Div(id="backtest-metrics", className="backtest-metrics")
            ], className="backtest-container"),
            
            # Auto-refresh
            dcc.Interval(
                id='interval-component',
                interval=self.config['visualization']['update_interval'] * 1000,
                n_intervals=0
            )
        ])
        
        # CSS personnalisé
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
            <head>
                {%metas%}
                <title>{%title%}</title>
                {%favicon%}
                {%css%}
                <style>
                    body {
                        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
                        margin: 0;
                        padding: 0;
                        background-color: #f5f5f5;
                    }
                    .header {
                        background-color: #1e2329;
                        color: white;
                        padding: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .header-title {
                        margin: 0;
                        font-size: 28px;
                    }
                    .header-subtitle {
                        font-size: 14px;
                        opacity: 0.8;
                        margin-top: 5px;
                    }
                    .controls {
                        display: flex;
                        align-items: center;
                        padding: 20px;
                        background-color: white;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin: 20px;
                        border-radius: 8px;
                    }
                    .control-item {
                        margin-right: 20px;
                    }
                    .button {
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        padding: 10px 20px;
                        cursor: pointer;
                        border-radius: 4px;
                        font-size: 16px;
                    }
                    .button:hover {
                        background-color: #45a049;
                    }
                    .charts-container {
                        margin: 20px;
                    }
                    .main-chart {
                        height: 500px;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin-bottom: 20px;
                    }
                    .indicator-row {
                        display: flex;
                        gap: 20px;
                        margin-bottom: 20px;
                    }
                    .indicator-chart {
                        flex: 1;
                        height: 250px;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .volume-chart {
                        height: 200px;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .tables-container {
                        display: flex;
                        gap: 20px;
                        margin: 20px;
                    }
                    .signals-container, .metrics-container {
                        flex: 1;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        padding: 20px;
                    }
                    .forecast-container, .backtest-container {
                        margin: 20px;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        padding: 20px;
                    }
                </style>
            </head>
            <body>
                {%app_entry%}
                <footer>
                    {%config%}
                    {%scripts%}
                    {%renderer%}
                </footer>
            </body>
        </html>
        '''
    
    def setup_callbacks(self):
        """Configure les callbacks du dashboard"""
        
        @self.app.callback(
            [Output('price-chart', 'figure'),
             Output('rsi-chart', 'figure'),
             Output('macd-chart', 'figure'),
             Output('volume-chart', 'figure'),
             Output('signals-table', 'children'),
             Output('metrics-table', 'children'),
             Output('forecast-chart', 'figure'),
             Output('backtest-chart', 'figure'),
             Output('last-update', 'children')],
            [Input('symbol-dropdown', 'value'),
             Input('date-range', 'start_date'),
             Input('date-range', 'end_date'),
             Input('refresh-button', 'n_clicks'),
             Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(symbol, start_date, end_date, n_clicks, n_intervals):
            """Met à jour tous les composants du dashboard"""
            
            # Charger les données
            data = self.load_symbol_data(symbol, start_date, end_date)
            
            # Créer les graphiques
            price_fig = self.create_price_chart(data, symbol)
            rsi_fig = self.create_rsi_chart(data, symbol)
            macd_fig = self.create_macd_chart(data, symbol)
            volume_fig = self.create_volume_chart(data, symbol)
            
            # Créer les tableaux
            signals_table = self.create_signals_table(data, symbol)
            metrics_table = self.create_metrics_table(data, symbol)
            
            # Créer les graphiques de prévision
            forecast_fig = self.create_forecast_chart(symbol)
            backtest_fig = self.create_backtest_chart(symbol)
            
            # Heure de dernière mise à jour
            last_update = f"Dernière mise à jour: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            
            return (price_fig, rsi_fig, macd_fig, volume_fig, 
                   signals_table, metrics_table, forecast_fig, 
                   backtest_fig, last_update)
    
    def create_price_chart(self, data, symbol):
        """Crée le graphique principal des prix avec indicateurs"""
        
        fig = go.Figure()
        
        # Chandeliers
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC'
        ))
        
        # Moyennes mobiles
        for ma in [20, 50, 200]:
            if f'MA{ma}' in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[f'MA{ma}'],
                    name=f'MA{ma}',
                    line=dict(width=1)
                ))
        
        # Bollinger Bands
        if 'BB_Upper' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_Upper'],
                name='BB Upper',
                line=dict(color='rgba(250, 128, 114, 0.3)', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['BB_Lower'],
                name='BB Lower',
                line=dict(color='rgba(250, 128, 114, 0.3)', width=1),
                fill='tonexty',
                fillcolor='rgba(250, 128, 114, 0.1)'
            ))
        
        # Support/Resistance
        if 'Support_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Support_20'],
                name='Support',
                line=dict(color='green', dash='dash', width=1)
            ))
        if 'Resistance_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Resistance_20'],
                name='Resistance',
                line=dict(color='red', dash='dash', width=1)
            ))
        
        fig.update_layout(
            title=f"{symbol} - Prix et Indicateurs Techniques",
            xaxis_title="Date",
            yaxis_title="Prix",
            template="plotly_white",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def create_forecast_chart(self, symbol):
        """Crée le graphique des prévisions multi-modèles"""
        
        # Charger les prévisions (simulé ici)
        # Dans la vraie implémentation, charger depuis les modèles
        
        fig = go.Figure()
        
        # Prix historiques récents
        # fig.add_trace(...)
        
        # Prévisions par modèle
        # for model in models:
        #     fig.add_trace(...)
        
        # Prévision d'ensemble
        # fig.add_trace(...)
        
        fig.update_layout(
            title=f"{symbol} - Prévisions Multi-Modèles",
            xaxis_title="Date",
            yaxis_title="Prix Prévu",
            template="plotly_white",
            height=400
        )
        
        return fig
    
    def run(self):
        """Lance le serveur du dashboard"""
        self.logger.info(f"Démarrage du dashboard sur le port {self.config['visualization']['port']}")
        self.app.run_server(debug=False, port=self.config['visualization']['port'])