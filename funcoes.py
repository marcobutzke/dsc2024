import pandas as pd
import numpy as np

from itertools import combinations
from statsmodels.stats.multicomp import pairwise_tukeyhsd

def cor_regiao(s):
    if s['Região'] == 'Norte':
        return ['background-color: beige']*len(s)
    elif s['Região'] == 'Nordeste':
        return ['background-color: gray']*len(s)
    elif s['Região'] == 'Centro Oeste':
        return ['background-color: lightgray']*len(s)
    elif s['Região'] == 'Sudeste':
        return ['background-color: lightblue']*len(s)
    elif s['Região'] == 'Sul':
        return ['background-color: lightgreen']*len(s)

def cor_classe(s):
    if s.outlier_max == 1:
        return ['background-color: #636efa']*len(s)
    elif s.class_lc == 'acima':
        return ['background-color: #00cc96']*len(s)
    elif s.class_lc == 'media':
        return ['background-color: #fecb52']*len(s)
    else:
        return ['background-color: #ef553b']*len(s)

def classificacao_estados_variavel(bra_var, variavel):
    iqr = bra_var[variavel].quantile(0.75) - bra_var[variavel].quantile(0.25)
    out_min = bra_var[variavel].quantile(0.25) - (1.5 * iqr)
    out_max = bra_var[variavel].quantile(0.75) + (1.5 * iqr)
    limite_inferior = bra_var[variavel].mean() - (1.96 * bra_var[variavel].std() / np.sqrt(len(bra_var)))
    limite_superior = bra_var[variavel].mean() + (1.96 * bra_var[variavel].std() / np.sqrt(len(bra_var)))
    bra_var['outlier_min'] = bra_var[variavel].apply(lambda x : 1 if x < out_min else 0)
    bra_var['outlier_max'] = bra_var[variavel].apply(lambda x : 1 if x > out_max else 0)
    bra_var['class_lc'] = bra_var[variavel].apply(
        lambda x : 'abaixo'
        if x < limite_inferior
        else (
            'acima'
            if x > limite_superior
            else 'media'
        )
    )
    return bra_var

def classificacao_abc_variavel(abc, variavel):
    abc['percentual'] = abc[variavel] / abc[variavel].sum()
    abc_sort = abc.sort_values(by=variavel, ascending=False).copy()
    abc_sort['percentual_acumulado'] = abc_sort['percentual'].cumsum()
    abc_sort['acumulado'] = abc_sort[variavel].cumsum()
    abc_sort['classe'] = abc_sort['percentual_acumulado'].apply(
        lambda x : 'A' if x <= 0.65 else ('B' if x <= 0.90 else 'C')
    )
    return abc_sort

def cor_abc(s):
    if s.classe == 'A':
        return ['background-color: #00cc96']*len(s)
    elif s.classe == 'B':
        return ['background-color: #fecb52']*len(s)
    else:
        return ['background-color: #ef553b']*len(s)

def analise_variancia(df, var):
    tukey = pairwise_tukeyhsd(
        endog=df[var], groups=df['Região'], alpha=0.05
    )
    tuk = []
    combinacao = combinations(tukey.groupsunique,2)
    for grupo in list(combinacao):
        tuk.insert(len(tuk), [grupo[0],grupo[1]])
    tuk = pd.DataFrame(tuk, columns=['grupo1', 'grupo2'])
    tuk['diferenca'] = tukey.reject
    tuk['valor'] = tukey.meandiffs
    return tuk
