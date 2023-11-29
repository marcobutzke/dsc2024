import pandas as pd
import numpy as np
import json
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_extras.metric_cards import style_metric_cards
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
import altair as alt
import folium
import yaml
from yaml.loader import SafeLoader

st.set_page_config(layout="wide")
style_metric_cards(
    border_left_color="#3D5077",
    background_color="#F0F2F6",
    border_size_px=3,
    border_color = "#CECED0",
    border_radius_px = 10,
    box_shadow=True
)

from funcoes import classificacao_estados_variavel, classificacao_abc_variavel, \
    cor_classe, cor_regiao, cor_abc, analise_variancia

@st.cache_data
def load_database_brasil(vars):
    bra = pd.read_excel('brasil_estados.xlsx')
    pad = bra[['Estado', 'Sigla', 'Região']].copy()
    for var in vars:
        variavel_media = bra[var].mean()
        variavel_desvio = bra[var].std()
        pad[var] = bra[var].apply(lambda x : ((x - variavel_media) / variavel_desvio) + 3)
    geo = json.load(open('brazil-states.geojson.txt'))
    return bra, pad, geo

@st.cache_data
def load_database_europa():
    return pd.read_parquet('SSEuropa.parquet')

with open('config.yalm') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(config['credentials'], config['cookie']['name'], config['cookie']['key'], config['cookie']['expiry_days'], config['preauthorized'])
authenticator.login('Login', 'main')
if st.session_state["authentication_status"]:
    authenticator.logout('Logout', 'main', key='unique_key')
    st.title(f'Aplicação: *{st.session_state["name"]}*')
    if st.session_state["name"] == 'Estados do Brasil':
        variaveis = ['População', 'Densidade Demográfica', 'Veículos', 'IDEB EF', 'IDEB EM',
                     'Rendimento per Capita', 'Ocupação ', 'Rendimento Ocupação', 'IDH', 'Área', 'Área Urbana']
        estados, padrao, geo_data = load_database_brasil(variaveis)
        dados, informacao, conhecimento = st.tabs(['Dados', 'Informação', 'Conhecimento/Inteligência'])
        with dados:
            if st.toggle('Visualizar por Região'):
                if st.toggle('Média dos Estados da Região'):
                    st.dataframe(
                        estados.groupby('Região')[variaveis].mean().reset_index().style.apply(cor_regiao, axis=1),
                        hide_index=True, use_container_width=True
                    )
                else:
                    st.dataframe(
                        estados.groupby('Região')[variaveis].sum().reset_index().style.apply(cor_regiao, axis=1),
                        hide_index=True, use_container_width=True
                    )
                regiao_cores = {'Nordeste':'gray', 'Sul':'lightgreen', 'Sudeste':'lightblue', 'Centro Oeste':'lightgray', 'Norte':'beige'}
                mapa_px = px.choropleth_mapbox(
                    data_frame = estados, geojson = geo_data, locations='Sigla',
                    featureidkey='properties.sigla', color='Região',
                    color_discrete_map=regiao_cores, mapbox_style='carto-positron', zoom=3.5,
                    center = {"lat": -15.76, "lon": -47.88}, opacity=1, width = 1200, height = 800,
                )
                mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
                mapa_px.update_traces(marker_line_width=1)
                st.plotly_chart(mapa_px)
            else:
                absoluto, padronizado = st.tabs(['Valores Absolutos', 'Valores Padronizados'])
                with absoluto:
                    st.dataframe(
                        estados[['Estado', 'Região'] + variaveis].style.apply(cor_regiao, axis=1),
                        hide_index=True, height=1000, use_container_width=True
                    )
                with padronizado:
                    st.dataframe(
                        padrao,
                        # padrao.style.apply(cor_regiao, axis=1),
                        hide_index=True, height=1000, use_container_width=True,
                        column_config={
                            "População": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                            "Densidade Demográfica": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                            "Veículos": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                            "IDEB EF": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                            "IDEB EM": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                            "Rendimento per Capita": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                            "Ocupação ": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                            "Rendimento Ocupação": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                            "IDH": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                            "Área": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                            "Área Urbana": st.column_config.ProgressColumn(width='small', format="%.2f", min_value=0, max_value=6),
                        },
                    )
        with informacao:
            variavel = st.selectbox('Selecione a coluna:', variaveis)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric('{0} Total'.format(variavel), estados[variavel].sum())
            c2.metric('{0} Média por Estados'.format(variavel), round(estados[variavel].mean(),2))
            c3.metric('{0} Máxima'.format(variavel), estados[variavel].max())
            c4.metric('{0} Mínima'.format(variavel), estados[variavel].min())
            if st.toggle('Curva ABC'):
                estados_abc = classificacao_abc_variavel(
                    estados[['Estado', 'Sigla', 'Região', variavel]].copy(),
                    variavel
                )
                tabl, mapa, graf = st.tabs(['Tabela', 'Mapas', 'Gráfico'])
                with tabl:
                    st.dataframe(estados_abc.style.apply(cor_abc, axis=1),
                                 hide_index=True, height=1000, use_container_width=True)
                with mapa:
                    minimo = estados[variavel].min()
                    maximo = estados[variavel].max()
                    mapa_px = px.choropleth_mapbox(
                        data_frame = estados, geojson = geo_data, locations='Sigla',
                        featureidkey='properties.sigla', color=variavel,
                        color_continuous_scale= 'blues', range_color=(minimo, maximo),
                        mapbox_style='carto-positron', zoom=2.5,
                        center = {"lat": -15.76, "lon": -47.88},
                        opacity=1, width = 640, height = 480,
                    )
                    mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
                    mapa_px.update_traces(marker_line_width=1)
                    cl1, cl2 = st.columns(2)
                    cl1.plotly_chart(mapa_px)
                    abc_cores = {'A':'#00cc96', 'B':'#fecb52', 'C':'#ef553b'}
                    mapa_px = px.choropleth_mapbox(
                        data_frame = estados_abc, geojson = geo_data, locations='Sigla',
                        featureidkey='properties.sigla', color='classe',
                        color_discrete_map=abc_cores, mapbox_style='carto-positron', zoom=2.5,
                        center = {"lat": -15.76, "lon": -47.88},
                        opacity=1, width = 640, height = 480,
                    )
                    mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
                    mapa_px.update_traces(marker_line_width=1)
                    cl2.plotly_chart(mapa_px)
                with graf:
                    sort_order = estados_abc["Estado"].tolist()
                    base = alt.Chart(estados_abc).encode(
                        x = alt.X("Estado:O",sort=sort_order),
                    ).properties(width = 1200, height=600)
                    bars = base.mark_bar(size = 15).encode(
                        y = alt.Y(variavel+':Q'), color = 'classe:N'
                    ).properties (width = 1200, height=600)
                    line = base.mark_line(strokeWidth= 1.5, color = "#cb4154").encode(
                        y=alt.Y('percentual_acumulado:Q', title='valores acumulados', axis=alt.Axis(format=".0%")), text = alt.Text('percentual_acumulado:Q'))
                    points = base.mark_circle(strokeWidth=3, color = "#cb4154").encode(
                            y=alt.Y('percentual_acumulado:Q', axis=None))
                    point_text = points.mark_text(align='left', baseline='middle', dx=-10, dy=-10).encode(
                        y= alt.Y('percentual_acumulado:Q', axis=None), text=alt.Text('percentual_acumulado:Q', format="0.0%"), color= alt.value("#cb4154"))
                    st.altair_chart((bars + line + points + point_text).resolve_scale(y = 'independent'))
            else:
                estados_cla = classificacao_estados_variavel(
                    estados[['Estado', 'Sigla', 'Região', variavel]].copy(),
                    variavel
                )
                tabl, mapa, graf = st.tabs(['Tabela', 'Mapas', 'Gráfico'])
                with tabl:
                    st.dataframe(estados_cla.style.apply(cor_classe, axis=1),
                                     hide_index=True, height=1000, use_container_width=True)
                with mapa:
                    cl1, cl2 = st.columns(2)
                    minimo = estados[variavel].min()
                    maximo = estados[variavel].max()
                    mapa_px = px.choropleth_mapbox(
                        data_frame = estados, geojson = geo_data, locations='Sigla',
                        featureidkey='properties.sigla', color=variavel,
                        color_continuous_scale= 'blues', range_color=(minimo, maximo),
                        mapbox_style='carto-positron', zoom=2.5,
                        center = {"lat": -15.76, "lon": -47.88},
                        opacity=1, width = 640, height = 480,
                    )
                    mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
                    mapa_px.update_traces(marker_line_width=1)
                    cl1.plotly_chart(mapa_px)
                    classe_cores = {'acima':'#00cc96', 'media':'#fecb52', 'abaixo':'#ef553b'}
                    mapa_px = px.choropleth_mapbox(
                        data_frame = estados_cla, geojson = geo_data, locations='Sigla',
                        featureidkey='properties.sigla', color='class_lc',
                        color_discrete_map=classe_cores, mapbox_style='carto-positron', zoom=2.5,
                        center = {"lat": -15.76, "lon": -47.88},
                        opacity=1, width = 640, height = 480,
                    )
                    mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
                    mapa_px.update_traces(marker_line_width=1)
                    cl2.plotly_chart(mapa_px)
                with graf:
                    st.altair_chart(alt.Chart(estados_cla).mark_bar().encode(
                        x="Estado:O", y=variavel+':Q', color='class_lc:N'
                    ).properties(height=600, width=1200))
            if st.toggle('Região'):
                col1, col2 = st.columns([1,2])
                regioes_diferentes = analise_variancia(estados, variavel)
                if col1.toggle('Mostrar todas as regiões'):
                    col1.dataframe(regioes_diferentes, hide_index=True)
                else:
                    col1.dataframe(regioes_diferentes[regioes_diferentes['diferenca'] == 1], hide_index=True)
                col2.plotly_chart(px.box(estados, x="Região", y=variavel))

        with conhecimento:
            cenario_vars = st.multiselect('Selecione o Cenário: ', variaveis)
            if len(cenario_vars) > 1:
                if st.toggle('Calcular Modelos'):
                    rnk, cls, emd, rlg, anm, vmp = st.tabs(
                        ['Ranking (classificação)', 'Grupo (Clusters)', 'Escalonamento Multidimensional',
                         'Região (Probabilidades)', 'Detecção de Anomalias', 'Associação']
                    )
                    with rnk:
                        ranking = padrao[['Estado', 'Sigla', 'Região'] + cenario_vars]
                        ranking['Score'] = padrao[cenario_vars].mean(axis=1)
                        ranking = classificacao_estados_variavel(ranking, 'Score')
                        tabl, mapa = st.tabs(['Tabela', 'Mapas'])
                        with tabl:
                            st.dataframe(ranking.sort_values(
                                by='Score', ascending=False
                            ).style.apply(cor_classe, axis=1),
                                hide_index=True, height=1000, use_container_width=True
                            )
                        with mapa:
                            c1, c2 = st.columns(2)
                            minimo = ranking['Score'].min()
                            maximo = ranking['Score'].max()
                            mapa_px = px.choropleth_mapbox(
                                data_frame = ranking, geojson = geo_data, locations='Sigla',
                                featureidkey='properties.sigla', color='Score',
                                color_continuous_scale= 'blues', range_color=(minimo, maximo),
                                mapbox_style='carto-positron', zoom=2.5,
                                center = {"lat": -15.76, "lon": -47.88},
                                opacity=1, width = 640, height = 480,
                            )
                            mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
                            mapa_px.update_traces(marker_line_width=1)
                            c1.plotly_chart(mapa_px)
                            classe_cores = {'acima':'#00cc96', 'media':'#fecb52', 'abaixo':'#ef553b'}
                            mapa_px = px.choropleth_mapbox(
                                data_frame = ranking, geojson = geo_data, locations='Sigla',
                                featureidkey='properties.sigla', color='class_lc',
                                color_discrete_map=classe_cores, mapbox_style='carto-positron', zoom=2.5,
                                center = {"lat": -15.76, "lon": -47.88},
                                opacity=1, width = 640, height = 480,
                            )
                            mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
                            mapa_px.update_traces(marker_line_width=1)
                            c2.plotly_chart(mapa_px)


    if st.session_state["name"] == 'Vendas na Europa':
        europa = load_database_europa()
        st.dataframe(europa)

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')


