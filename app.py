from numpy.core.fromnumeric import size
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pydicom as dc
import math
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


st.set_page_config(layout="wide")


def S(x, alfa, beta): return np.exp(-alfa*x-beta*x*x)
def DBE(n,d,alfabeta): return n*d*(1+d/alfabeta)


def EQDX(alfabeta, dosistotal, dosisref,dosis): return (
    dosis+alfabeta)/(dosisref+alfabeta)

def loaddata(file):
    return pd.read_csv(file,sep=';',decimal=',')


Prostata=loaddata('Prostata.csv')


st.title('El modelo lineal cuadrático')

st.markdown('## 1. Curvas de supervivencia')

st.latex(
    r'''S(d)= e^{-\alpha d-\beta d^2}''')

c1_alfa = st.slider('\u03B1', min_value=0.0,
                    max_value=10.0, value=0.1, step=0.05)
c1_beta = st.slider('\u03B2', min_value=0.0,
                    max_value=10.0, value=0.1, step=0.05)

st.write('\u03B1/\u03B2 ', np.round(c1_alfa/c1_beta, decimals=2))

c1_Supervivencia = pd.DataFrame()
c1_Supervivencia['Dosis'] = np.linspace(0, 12, num=120)
c1_Supervivencia['Supervivencia'] = S(
    c1_Supervivencia['Dosis'], c1_alfa, c1_beta)

fig = px.line(c1_Supervivencia, x="Dosis", y="Supervivencia",
              log_y=True, range_y=[1e-10, 1])
st.plotly_chart(fig)


st.markdown('## 2. Fraccionamiento')

st.latex(
    r'''S(D)=(S(d))^n= e^{(-\alpha d-\beta d^2) \cdot n}=e^{-\alpha D-\frac{1}{n}\beta D^2}''')

c2_alfa = st.slider('\u03B1', min_value=0.0, max_value=10.0,
                    value=0.1, step=0.05, key='c2_alfa')
c2_beta = st.slider('\u03B2', min_value=0.0, max_value=10.0,
                    value=0.1, step=0.05, key='c2_beta')

st.write('\u03B1/\u03B2 ', np.round(c2_alfa/c2_beta, decimals=2))

c2_col1, c2_col2, c2_col3 = st.columns(3)
with c2_col1:
    c2_d = st.number_input('Dosis por fracción (Gy)',
                           min_value=0.0, value=1.0, step=0.5, key='c2_d')
with c2_col2:
    c2_n1 = st.number_input('Fraccionamiento 1',
                            min_value=1, step=1, key='c2_n1')
with c2_col3:
    c2_n2 = st.number_input('Fraccionamiento 2',
                            min_value=1, step=1, key='c2_n2')


c2_Supervivencia_1 = pd.DataFrame()

c2_Supervivencia_1['Dosis'] = np.linspace(
    0, float(c2_d*c2_n1), num=int(c2_d*c2_n1/0.1))
c2_Supervivencia_1['Fraccionado'] = S(
    c2_Supervivencia_1['Dosis'], c2_alfa, c2_beta/c2_n1)
c2_Supervivencia_1['Una fracción'] = S(
    c2_Supervivencia_1['Dosis'], c2_alfa, c2_beta)


c2_fig_1 = px.line(c2_Supervivencia_1, x="Dosis", y=[
                   "Fraccionado", "Una fracción"], log_y=True, range_y=[1e-10, 1])

c2_Supervivencia_2 = pd.DataFrame()

c2_Supervivencia_2['Dosis'] = np.linspace(
    0, float(c2_d*c2_n2), num=int(c2_d*c2_n2/0.1))
c2_Supervivencia_2['Fraccionado']=S(
    c2_Supervivencia_2['Dosis'], c2_alfa, c2_beta/c2_n2)
c2_Supervivencia_2['Una fracción']=S(
    c2_Supervivencia_2['Dosis'], c2_alfa, c2_beta)


c2_col_fig1, c2_col_fig2=st.columns([1, 1])
with c2_col_fig1:
    c2_fig_1=px.line(c2_Supervivencia_1, x = "Dosis", y = [
                   "Fraccionado", "Una fracción"], log_y = True, range_y = [1e-10, 1])
    st.plotly_chart(c2_fig_1)
with c2_col_fig2:
    c2_fig_2=px.line(c2_Supervivencia_2, x = "Dosis", y = [
        "Fraccionado", "Una fracción"], log_y = True, range_y = [1e-10, 1])
    st.plotly_chart(c2_fig_2)

st.markdown('## 3. Dosis biologicamente equivalente (DBE)')

st.latex(
    r'''DBE=nd \left ( 1+\frac{d}{\alpha/\beta} \right )''')

c3_alfabeta_respuestaprecoz = st.slider('\u03B1 /\u03B2 respuesta precoz',min_value=1,max_value=15,value=3)
c3_alfabeta_respuestatardia = st.slider('\u03B1 /\u03B2 respuesta tardía',min_value=1,max_value=15,value=10)
c3_dosis = st.slider('Selecciona el rango de dosis',0.0, 30.0, (2.0, 6.0))
c3_fracciones = st.slider('Selecciona el rango de fracciones',1, 40,(2, 6),step=1)

c3_DBE_1 = pd.DataFrame()

c3_DBE_1['Dosis'] = np.linspace(
    c3_dosis[0], float(c3_dosis[1]), num=int((c3_dosis[1]-c3_dosis[0])/0.1))
c3_DBE_1['DBE precoz'] = DBE(
    c3_fracciones[0], c3_DBE_1['Dosis'], c3_alfabeta_respuestaprecoz)
c3_DBE_1['DBE tardía'] = DBE(
    c3_fracciones[0], c3_DBE_1['Dosis'], c3_alfabeta_respuestatardia)
c3_fig_1 = px.line(c3_DBE_1, x="Dosis", y=[
    "DBE precoz", "DBE tardía"])

c3_fig_1.update_layout(yaxis_title='DBE')

full_fig = c3_fig_1.full_figure_for_development()

c3_DBE_2 = pd.DataFrame()

c3_DBE_2['Dosis'] = np.linspace(
    c3_dosis[0], float(c3_dosis[1]), num=int((c3_dosis[1]-c3_dosis[0])/0.1))
c3_DBE_2['DBE precoz'] = DBE(
    c3_fracciones[1], c3_DBE_2['Dosis'], c3_alfabeta_respuestaprecoz)
c3_DBE_2['DBE tardía'] = DBE(
    c3_fracciones[1], c3_DBE_2['Dosis'], c3_alfabeta_respuestatardia)
c3_fig_2 = px.line(c3_DBE_2, x="Dosis", y=[
    "DBE precoz", "DBE tardía"], range_y=full_fig.layout.yaxis.range)
c3_fig_2.update_layout(yaxis_title='DBE')


c3_col_fig1, c3_col_fig2 = st.columns([1, 1])
with c3_col_fig1:
    st.plotly_chart(c3_fig_1)
with c3_col_fig2:
    st.plotly_chart(c3_fig_2)


st.markdown('## 4. Histogramas Dosis-Volumen')


c4_DosisTotal=6000
c4_DosisRef=3.00
c4_alfabeta_sanos = st.slider('\u03B1 /\u03B2 respuesta precoz',min_value=1,max_value=15,value=3,key='c4_alfabeta_sanos')
c4_alfabeta_tumoral = st.slider(
    '\u03B1 /\u03B2 respuesta tardía', min_value=1, max_value=15, value=10, key='c4_alfabeta_respuestardia')
c4_dosis = st.slider('Selecciona la dosis',0.0, 10.0, value=3.0,key='c4_dosis')
st.write('Dosis de referencia 6000 cGy A 300 cGy por fracción')

c4_DBE = Prostata
c4_DBE['Dosis_sanos'] = EQDX(
    c4_alfabeta_sanos, c4_DosisTotal, c4_DosisRef, c4_dosis)*c4_DBE['Dosis']
c4_DBE['Dosis_tumoral'] = EQDX(
    c4_alfabeta_tumoral, c4_DosisTotal, c4_DosisRef, c4_dosis)*c4_DBE['Dosis']

c4_fig_1 = px.line(Prostata, x="Dosis", y=["PTV","Bufeta","Recto"],width=500)
c4_fig_2 = px.line(Prostata, x="Dosis_sanos",
                   y=["PTV", "Bufeta", "Recto"], width=500)
c4_fig_3 = px.line(
    Prostata, x="Dosis_tumoral", y=["PTV", "Bufeta", "Recto"], width=500)

c4_col_fig1, c4_col_fig2,c4_col_fig3=st.columns([1, 1,1])
with c4_col_fig1:
    st.plotly_chart(c4_fig_1)
with c4_col_fig2:
    st.plotly_chart(c4_fig_2)
with c4_col_fig3:
    st.plotly_chart(c4_fig_3)    

st.markdown('## 5. Isoefecto: Hiperfraccionamiento')

st.latex(
    r'''DBE=nd \left ( 1+\frac{d(1+h_m)}{\alpha/\beta} \right )''')

st.latex(
    r'''h_m =\left(frac{2}{m} \right)\frac{\theta}{(1 -\theta)}\left[m -\frac{(1 -\theta ^ {m})}{(1 -\theta)} \right]''')



