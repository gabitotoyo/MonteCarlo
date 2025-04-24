from flask import Flask, render_template, request
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

app = Flask(__name__)

# Configuración API
API_BASE = "https://api.cambiocuba.money/api/v1/x-rates-by-date-range-history"
PARAMS = {
    "trmi": "true",
    "cur": "USD",
    "date_from": "2021-01-01 00:00:00"
}

def obtener_datos_actuales():
    """Obtiene y procesa datos de la API"""
    try:
        fecha_hasta = datetime.now().strftime("%Y-%m-%d 23:59:59")
        params = {**PARAMS, "date_to": fecha_hasta}
        response = requests.get(API_BASE, params=params)
        response.raise_for_status()
        datos = response.json()
        return procesar_datos_crudos(datos)
    except Exception as e:
        print(f"Error API: {e}")
        return pd.DataFrame()

def procesar_datos_crudos(datos_api):
    """Procesa datos crudos"""
    registros = []
    for item in datos_api:
        try:
            fecha = item.get("_id", "")
            median = str(item.get("median", "")).split("JS:")[0].strip()
            registros.append({
                "Fecha": pd.to_datetime(fecha),
                "CUPs": float(median) if median.replace('.','',1).isdigit() else None
            })
        except (ValueError, AttributeError):
            continue
    
    if registros:
        df = pd.DataFrame(registros).sort_values('Fecha')
        df.set_index('Fecha', inplace=True)
        return df
    return pd.DataFrame()

def calcular_indicadores(df):
    """Calcula indicadores técnicos"""
    if df.empty:
        return df
    
    # Medias móviles
    df['SMA30'] = df['CUPs'].rolling(30, min_periods=1).mean()
    df['SMA200'] = df['CUPs'].rolling(200, min_periods=1).mean()
    
    # Bollinger Bands
    df['SMA20'] = df['CUPs'].rolling(20).mean()
    df['UpperBB'] = df['SMA20'] + (2 * df['CUPs'].rolling(20).std())
    df['LowerBB'] = df['SMA20'] - (2 * df['CUPs'].rolling(20).std())
    
    # Señales de trading
    df['Cruce'] = np.where(df['SMA30'] > df['SMA200'], 1, -1)
    df['Señal'] = df['Cruce'].diff()
    
    return df.dropna()

def simular_montecarlo(df, capital=1000, meses=1, simulaciones=10**10):
    """Realiza simulación de Montecarlo para múltiples meses"""
    if df.empty or len(df) < 2:
        return None
    
    # Calcular retornos mensuales
    mensual = df.resample('M').last()
    retornos = mensual['CUPs'].pct_change().dropna()
    
    if len(retornos) < 2:
        return None
    
    mu = retornos.mean()
    sigma = retornos.std()
    
    # Simulación para múltiples meses
    np.random.seed(42)
    sim_retornos = np.zeros(simulaciones)
    
    for _ in range(meses):
        monthly_returns = np.random.normal(mu, sigma, simulaciones)
        sim_retornos = (1 + sim_retornos) * (1 + monthly_returns) - 1
    
    # Probabilidades
    prob_subida = (sim_retornos > 0).mean() * 100
    prob_bajada = 100 - prob_subida
    
    # Cálculo de inversión
    if prob_subida > 60:
        inversion = 0.7 * capital
        accion = "COMPRAR"
    elif prob_bajada > 40:
        inversion = -0.5 * capital
        accion = "VENDER"
    else:
        inversion = ((prob_subida/100 - 0.5)/0.5) * capital
        accion = "COMPRAR" if inversion > 0 else "VENDER"
    
    # Rentabilidad esperada
    rent_esperada = np.mean(sim_retornos) * 100
    utilidades = capital * (rent_esperada / 100)
    
    # Calculo de riesgos
    var_95 = np.percentile(sim_retornos, 5) * 100
    mejor_escenario = np.percentile(sim_retornos, 95) * 100
    
    return {
        'precio_actual': df.iloc[-1]['CUPs'],
        'prob_subida': prob_subida,
        'prob_bajada': prob_bajada,
        'accion': accion,
        'inversion': abs(inversion),
        'rent_esperada': rent_esperada,
        'utilidades': utilidades,
        'var_95': var_95,
        'mejor_escenario': mejor_escenario,
        'mu': mu * 100,
        'sigma': sigma * 100,
        'meses': meses
    }

def generar_grafico_plotly(df):
    """Crea gráfico con Bollinger Bands y señales"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=("Evolución del Tipo de Cambio",)
    )
    
    # Precio y Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['CUPs'], 
            name="Precio", 
            line=dict(color='#1f77b4'),
        showlegend=False),
        row=1, col=1)
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['UpperBB'],
            name="Banda Superior",
            line=dict(color='rgba(255, 0, 0, 0.3)', width=1),
            showlegend=False),
        row=1, col=1)
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['LowerBB'],
            name="Banda Inferior",
            line=dict(color='rgba(0, 255, 0, 0.3)', width=1),
            fill='tonexty',
            fillcolor='rgba(100, 100, 100, 0.1)',
            showlegend=False),
        row=1, col=1)
    
    # Medias móviles
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['SMA30'], 
            name="SMA 30", 
            line=dict(dash='dot', color='orange'),
            showlegend=False),
        row=1, col=1)
    
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['SMA200'], 
            name="SMA 200", 
            line=dict(dash='dot', color='purple'), 
            showlegend=False
        ),
        row=1, col=1)
    
    # Señales de compra/venta
    compras = df[df['Señal'] == 2]
    ventas = df[df['Señal'] == -2]
    
    fig.add_trace(
        go.Scatter(
            x=compras.index,
            y=compras['CUPs'],
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Señal Compra',
            showlegend=False
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=ventas.index,
            y=ventas['CUPs'],
            mode='markers',
            marker=dict(symbol='triangle-down', size=10, color='red'),
            name='Señal Venta',
            showlegend=False
        )
    )
    
    fig.update_layout(
        height=600,
        title_text="Análisis Técnico del Dólar Informal",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig.to_html(full_html=False)

@app.route("/", methods=['GET', 'POST'])
def home():
    """Endpoint principal"""
    capital = 1000  # Valores por defecto
    meses = 1
    
    if request.method == 'POST':
        capital = float(request.form.get('capital', 1000))
        meses = int(request.form.get('meses', 1))
    
    df = obtener_datos_actuales()
    
    if not df.empty:
        df = calcular_indicadores(df)
        analisis = simular_montecarlo(df, capital, meses)
        grafico = generar_grafico_plotly(df)
    else:
        analisis = None
        grafico = "<p>Error cargando datos</p>"
    
    return render_template(
        "index.html",
        grafico=grafico,
        analisis=analisis,
        capital=capital,
        meses=meses
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
