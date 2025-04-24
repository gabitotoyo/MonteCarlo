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

def simular_montecarlo(df, capital=1000, simulaciones=10**6):
    """Realiza simulación de Montecarlo y calcula recomendación"""
    if df.empty or len(df) < 2:
        return None
    
    # Calcular retornos mensuales
    mensual = df.resample('M').last()
    retornos = mensual['CUPs'].pct_change().dropna()
    
    if len(retornos) < 2:
        return None
    
    mu = retornos.mean()
    sigma = retornos.std()
    
    # Simulación
    np.random.seed(42)
    sim_retornos = np.random.normal(mu, sigma, simulaciones)
    
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
    rent_esperada = np.mean(sim_retornos) * abs(inversion)/capital * 100
    
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
        'var_95': var_95,
        'mejor_escenario': mejor_escenario,
        'mu': mu * 100,
        'sigma': sigma * 100
    }

def generar_grafico_plotly(df):
    """Crea gráficos interactivos"""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=("Evolución del Tipo de Cambio", "Distribución Histórica"),
        row_heights=[0.7, 0.3]
    )
    
    # Gráfico principal
    fig.add_trace(
        go.Scatter(
            x=df.index, 
            y=df['CUPs'], 
            name="Precio", 
            line=dict(color='#1f77b4')),
        row=1, col=1)
    
    # Histograma
    fig.add_trace(
        go.Histogram(
            x=df['CUPs'], 
            name="Distribución", 
            marker_color='#2ca02c',
            nbinsx=50),
        row=2, col=1)
    
    fig.update_layout(
        height=800,
        title_text="Análisis del Dólar Informal en Cuba",
        template="plotly_white"
    )
    
    return fig.to_html(full_html=False)

@app.route("/", methods=['GET', 'POST'])
def home():
    """Endpoint principal"""
    capital = 1000  # Valor por defecto
    
    if request.method == 'POST':
        capital = float(request.form.get('capital', 1000))
    
    df = obtener_datos_actuales()
    
    if df.empty:
        return render_template("index.html", grafico="<p>Error cargando datos</p>")
    
    analisis = simular_montecarlo(df, capital)
    grafico = generar_grafico_plotly(df)
    
    return render_template(
        "index.html",
        grafico=grafico,
        analisis=analisis,
        capital=capital
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)