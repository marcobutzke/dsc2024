import pandas as pd
import numpy as np
import json
import streamlit as st
from streamlit_extras.metric_cards import style_metric_cards
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import folium_static
import altair as alt
import folium
from folium.plugins import MarkerCluster
import branca.colormap as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from prophet import Prophet

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
    cor_classe, cor_regiao, cor_abc, analise_variancia, cor_cluster

@st.cache_data
def load_database_brasil(vars):
    bra = pd.read_parquet('brasil_estados.parquet')
    pad = bra[['Estado', 'Sigla', 'Região']].copy()
    for var in vars:
        variavel_media = bra[var].mean()
        variavel_desvio = bra[var].std()
        pad[var] = bra[var].apply(lambda x : ((x - variavel_media) / variavel_desvio) + 3)
    geo = json.load(open('brazil-states.geojson.txt'))
    return bra, pad, geo

@st.cache_data
def load_database_europa():
    europe = pd.read_parquet('SSEuropa.parquet')
    europe['Year'] = europe['Order Date'].dt.year
    europe['Month'] = europe['Order Date'].dt.month
    europe['Order Date Month'] = europe['Order Date'].apply(lambda x : x.strftime("%Y-%m-01"))
    return europe

variaveis = ['População', 'Densidade Demográfica', 'Veículos', 'IDEB EF', 'IDEB EM',
         'Rendimento per Capita', 'Ocupação ', 'Rendimento Ocupação', 'IDH', 'Área', 'Área Urbana']
estados, padrao, geo_data = load_database_brasil(variaveis)
europa = load_database_europa()

estd, euro = st.tabs(['Estados do Brasil', 'Vendas na Europa'])

with estd:
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
                rnk, cls, emd, anm, vmp = st.tabs(
                    ['Ranking (classificação)', 'Grupo (Clusters)', 'Escalonamento Multidimensional',
                     'Detecção de Anomalias', 'Associação']
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
                with cls:
                    kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(estados[cenario_vars])
                    estados_cls = estados[['Estado', 'Sigla', 'Região'] + cenario_vars]
                    estados_cls['Cluster'] = kmeans.labels_
                    st.dataframe(estados_cls.pivot_table(
                        index='Cluster', values=cenario_vars, aggfunc='mean'
                    ).reset_index().style.apply(cor_cluster, axis=1), hide_index=True, use_container_width=True)
                    tabl, mapa = st.tabs(['Tabela', 'Mapas'])
                    with tabl:
                        st.dataframe(estados_cls.sort_values(by='Cluster').style.apply(cor_cluster, axis=1),
                            hide_index=True, height=1000, use_container_width=True
                        )
                    with mapa:
                        clusters_cores = {0:'beige', 1:'gray', 2:'lightgray', 3:'lightblue', 4:'lightgreen'}
                        mapa_px = px.choropleth_mapbox(
                            data_frame = estados_cls, geojson = geo_data, locations='Sigla',
                            featureidkey='properties.sigla', color='Cluster',
                            color_discrete_map=clusters_cores, mapbox_style='carto-positron', zoom=2.5,
                            center = {"lat": -15.76, "lon": -47.88},
                            opacity=1, width = 640, height = 480,
                        )
                        mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
                        mapa_px.update_traces(marker_line_width=1)
                        st.plotly_chart(mapa_px)
                with emd:
                    pca = PCA(n_components=2).fit(estados[cenario_vars]).transform(estados[cenario_vars])
                    estados_emd = estados[['Estado', 'Sigla', 'Região'] + cenario_vars].copy()
                    estados_emd['escalaX'] = pca[:, 0]
                    estados_emd['escalaY'] = pca[:, 1]
                    kmeans = KMeans(n_clusters=5, random_state=0, n_init='auto').fit(estados[cenario_vars])
                    estados_emd['Cluster'] = kmeans.labels_
                    st.dataframe(estados_emd.pivot_table(
                        index='Cluster', values=cenario_vars, aggfunc='mean'
                    ).reset_index(), hide_index=True, use_container_width=True)
                    ed = alt.Chart(estados_emd).mark_circle(size=100).encode(
                        alt.X('escalaX', scale=alt.Scale(zero=False)),
                        alt.Y('escalaY', scale=alt.Scale(zero=False, padding=1)),
                        color='Cluster:N',
                    ).encode(tooltip=['Estado', 'Sigla']).properties(width=1200, height=800)
                    tx = alt.Chart(estados_emd).mark_text(dy=-10).encode(
                        alt.X('escalaX', scale=alt.Scale(zero=False)),
                        alt.Y('escalaY', scale=alt.Scale(zero=False, padding=1)),
                        text = 'Estado'
                    ).encode(tooltip=['Estado', 'Sigla']).properties(width=1200, height=800)
                    st.altair_chart(ed + tx)
                with anm:
                    clf = KNN().fit(StandardScaler().fit_transform(estados[cenario_vars]))
                    estados_anm = estados[['Estado', 'Sigla', 'Região'] + cenario_vars].copy()
                    estados_anm['Outlier'] = clf.predict(StandardScaler().fit_transform(estados[cenario_vars]))
                    st.dataframe(estados_anm[estados_anm['Outlier'] == 1],
                                 hide_index=True, use_container_width=True)
                    outliers_cores = {0:'white', 1:'blue'}
                    mapa_px = px.choropleth_mapbox(
                        data_frame = estados_anm, geojson = geo_data, locations='Sigla',
                        featureidkey='properties.sigla', color='Outlier',
                        color_discrete_map=outliers_cores, mapbox_style='carto-positron', zoom=2.5,
                        center = {"lat": -15.76, "lon": -47.88},
                        opacity=1, width = 640, height = 480,
                    )
                    mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
                    mapa_px.update_traces(marker_line_width=1)
                    st.plotly_chart(mapa_px)
                with vmp:
                    estado = st.selectbox('Selecione o Estado:', estados['Estado'])
                    estado_vmp = estados[estados['Estado'] == estado][cenario_vars]
                    estados_vmp = estados[cenario_vars]
                    neighbors_alg = NearestNeighbors(n_neighbors=min(6, len(estados_vmp))).fit(estados_vmp)
                    similar = neighbors_alg.kneighbors(estado_vmp, return_distance=False)[0]
                    estados_similares = list(estados.iloc[similar]['Sigla'])
                    estados_knn = estados[['Estado', 'Sigla', 'Região'] + cenario_vars].copy()
                    estados_knn['Similar'] = estados_knn.apply(lambda x : 1 if x['Sigla'] in estados_similares else 0, axis=1)
                    st.dataframe(estados_knn[estados_knn['Similar'] == 1],
                                 hide_index=True, use_container_width=True)
                    similar_cores = {0:'white', 1:'blue'}
                    mapa_px = px.choropleth_mapbox(
                        data_frame = estados_knn, geojson = geo_data, locations='Sigla',
                        featureidkey='properties.sigla', color='Similar',
                        color_discrete_map=similar_cores, mapbox_style='carto-positron', zoom=2.5,
                        center = {"lat": -15.76, "lon": -47.88},
                        opacity=1, width = 640, height = 480,
                    )
                    mapa_px.update_layout(margin={'r':0,'t':0,'l':0, 'b':0})
                    mapa_px.update_traces(marker_line_width=1)
                    st.plotly_chart(mapa_px)
with euro:
    ped, seg, luc, prv, prj = st.tabs(
        [
            'Localização dos Pedidos',
            'Segmentos do Mercado',
            'Lucratividade por Localização',
            'Previsão de Lucro por Produto',
            'Projeção de Vendas/Lucro'
        ]
    )
    with ped:
        c1, c2, c3, c4 = st.columns([2,2,1,1])
        linhas = c1.multiselect(
            'Linha(s):', ['Country', 'State/Province', 'City', 'Sub-Category', 'Product', 'Trademark']
        )
        colunas = c2.multiselect(
            'Coluna(s):', ['Year', 'Month', 'Category', 'Segment', 'Region']
        )
        valores = c3.selectbox('Valor:', ['Sales', 'Profit', 'Quantity'])
        agregador = c4.selectbox('Função:', ['sum', 'mean', 'count', 'min', 'max'])
        if len(linhas) > 0 or len(colunas)> 0:
            st.dataframe(
                europa.pivot_table(
                    index=linhas, columns=colunas, values=valores, aggfunc=agregador, fill_value=0
                ), use_container_width=True
            )
        if st.toggle('Mapa'):
            europa_cid = europa.groupby(['City', 'Lat', 'Lng'])[
                ['Sales','Profit','Quantity']
            ].sum().reset_index()
            euromapc = folium.Map(location=[europa_cid['Lat'].mean(), europa_cid['Lng'].mean()],
                                  tiles='cartodbpositron', zoom_start=4,
                                  width=800, height=600)
            mapc = MarkerCluster()
            for idx, row in europa_cid.iterrows():
                mapc.add_child(folium.Marker([row['Lat'], row['Lng']],popup=row['City']))
            euromapc.add_child(mapc)
            folium_static(euromapc)
    with seg:
        segmento = st.selectbox('Segmento: ', europa['Segment'].unique())
        europa_seg = europa[europa['Segment'] == segmento].groupby(
            ['Region', 'Country', 'State/Province', 'City', 'Lat', 'Lng']
        )[['Sales','Profit','Quantity']].sum().reset_index()
        media = europa_seg['Sales'].mean()
        dp = europa_seg['Sales'].std()
        europa_seg['Rateio'] = europa_seg['Sales'].apply(lambda x : int((x - media) / dp) + 3)
        mapseg = folium.Map(location=[europa_seg['Lat'].mean(), europa_seg['Lng'].mean()],
                            tiles='cartodbpositron', zoom_start=4,
                            width=800, height=600)
        region_cores = {'South': 'red', 'North': 'blue', 'Central':'green'}
        for idx, row in europa_seg.iterrows():
            mapseg.add_child(folium.CircleMarker(
                [row['Lat'], row['Lng']],
                popup='Country: {0} State/Province: {1} - City: {2}'.format(
                    row['Country'],row['State/Province'],row['City']),
                radius=row['Rateio'], color=region_cores[row['Region']]
                )
            )
        folium_static(mapseg)
    with luc:
        europa_prb = europa.groupby(
            ['Region', 'Country', 'State/Province', 'City', 'Lat', 'Lng']
        )[['Sales','Profit','Quantity']].mean()
        media = europa_prb['Profit'].mean()
        europa_prb['Esperado'] = europa_prb['Profit'].apply(lambda x : 1 if x > media else 0)
        X_Train = europa_prb.drop(columns=['Esperado'], axis=1)
        X_Test = europa_prb.drop(columns=['Esperado'], axis=1)
        y_Train = europa_prb['Esperado']
        y_Test = europa_prb['Esperado']
        sc_x = StandardScaler()
        X_Train = sc_x.fit_transform(X_Train)
        X_Test = sc_x.fit_transform(X_Test)
        logreg = LogisticRegression(solver="lbfgs", max_iter=500)
        logreg.fit(X_Train, y_Train)
        pred_logreg = logreg.predict(X_Test)
        europa_prb['Previsao'] = pred_logreg
        pred_proba = logreg.predict_proba(X_Test)
        lista_proba = pred_proba.tolist()
        lista_proba = pd.DataFrame(
            lista_proba, columns = ['Prob_prejuizo', 'Prob_lucro']
        )
        europa_prb = europa_prb.reset_index()
        europa_prb = pd.merge(europa_prb, lista_proba, left_index=True, right_index=True)
        linear = cm.LinearColormap(["red", "yellow", "green"],
                                   vmin=europa_prb['Prob_lucro'].min(),
                                   vmax=europa_prb['Prob_lucro'].max())
        mapprb = folium.Map(
            [europa_prb['Lat'].mean(), europa_prb['Lng'].mean()],
            tiles="cartodbpositron", zoom_start=4,
            width=800, height=600)
        for index, row in europa_prb.iterrows():
            folium.CircleMarker([row['Lat'], row['Lng']],
                              popup='Probabilidade Esperada de Lucro acima da média: {0}'.format(
                                  round(row['Prob_lucro'],4)),
                              radius=3,
                              color = linear(row['Prob_lucro']),
                              ).add_to(mapprb)
        c1, c2 = st.columns([1,2])
        c1.dataframe(europa_prb.groupby('Country')['Prob_lucro'].mean().reset_index().sort_values(
            by='Prob_lucro', ascending=False
        ), hide_index=True, use_container_width=True, height=600)
        with c2:
            folium_static(mapprb)
    with prv:
        regressao = europa.groupby(['Category','Sub-Category','Product'])[['Profit']].mean().reset_index()
        regressao = regressao.merge(
        europa.pivot_table(
            index=['Category','Sub-Category','Product'], columns='Segment', values='Sales', aggfunc='sum', fill_value=0).reset_index(),
                                     on=['Category','Sub-Category','Product'], how='left')
        regressao = regressao.merge(
        europa.pivot_table(
            index=['Category','Sub-Category','Product'], columns='Region', values='Sales', aggfunc='sum', fill_value=0).reset_index(),
                                     on=['Category','Sub-Category','Product'], how='left')
        regr = regressao[['Profit','Consumer','Corporate','Home Office', 'Central', 'North', 'South']]
        X_Train = regr.drop(columns=['Profit'], axis=1)
        X_Test = regr.drop(columns=['Profit'], axis=1)
        y_Train = regr['Profit']
        y_Test = regr['Profit']
        rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        rf.fit(X_Train, y_Train)
        regressao['Esperado'] = rf.predict(X_Test)
        regressao['Diferenca'] = regressao['Profit'] - regressao['Esperado']
        c1, c2 = st.columns(2)
        c1.metric('Valor Médio Lucro Obtido', round(regressao['Profit'].mean(),2))
        c1.dataframe(
            regressao.groupby('Category')[['Profit','Esperado','Diferenca']].mean().reset_index().sort_values(
                by='Diferenca', ascending=False
            ),
            hide_index=True, use_container_width=True)
        c1.dataframe(
            regressao.groupby(
                ['Category', 'Sub-Category'])[['Profit','Esperado','Diferenca']].mean().reset_index().sort_values(
                by='Diferenca', ascending=False
            ),
            hide_index=True, use_container_width=True, height=640)
        c2.metric('Valor Médio Lucro Esperado', round(regressao['Esperado'].mean(),2))
        c2.dataframe(
            regressao[
                ['Category','Sub-Category','Product','Profit','Esperado','Diferenca']].sort_values(
                by='Diferenca', ascending=False
            ),
            hide_index=True, use_container_width=True, height=1200)
    with prj:
        europa_sls = europa.groupby('Order Date Month')[['Sales']].sum().reset_index()
        europa_sls = europa_sls.rename(columns={'Order Date Month': 'ds', 'Sales': 'y'})
        euromdl = Prophet().fit(europa_sls)
        future = euromdl.make_future_dataframe(periods=12, freq='MS')
        forecast = euromdl.predict(future)
        c1, c2 = st.columns(2)
        c1.header('Vendas')
        c1.pyplot(euromdl.plot(forecast))
        c1.dataframe(forecast.tail(12)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                     hide_index=True, use_container_width=True, height=480)
        europa_prf = europa.groupby('Order Date Month')[['Profit']].sum().reset_index()
        europa_prf = europa_prf.rename(columns={'Order Date Month': 'ds', 'Profit': 'y'})
        euromdl = Prophet().fit(europa_prf)
        future = euromdl.make_future_dataframe(periods=12, freq='MS')
        forecast = euromdl.predict(future)
        c2.header('Lucro')
        c2.pyplot(euromdl.plot(forecast))
        c2.dataframe(forecast.tail(12)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
                     hide_index=True, use_container_width=True, height=480)
