import streamlit as st

from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

def pickle_var(path,var=None):
    import pickle

    #complex types as np.array or pandas DataFrame don't allow a simple comparison with None
    if type(var)==type(None):
         with open(path,'rb') as fich:
            return pickle.load(fich)

    else:
        with open(path,'wb') as fich:
            pickle.dump(var,fich)
# COMPREL routines

def plot_serie_conprel(df,labelx=None,labely=None):
    graph=sns.lineplot(data=df)
    ymax=np.ceil(max(df.max()))
    ymin=np.floor(min(df.min()))
    for elec in list(range(2011,int(df.index.max())+1,4)):
        if elec>=df.index.min():
            plt.vlines(elec,ymin,ymax,color='red',linestyle=':')

    if labelx:
        plt.xlabel(labelx)
    if labely:
        plt.ylabel(labely)
    plt.grid()
    return graph

def create_legend_conprel(df):
    legend={}
    for col in df.columns:
        sigla=''.join([item[0].upper() for item in col.split() if len(item)>3])

        index=1
        while sigla in legend.values():
            sigla+=col.split()[-1][index]
            index+=1
        legend[col]=sigla
    return legend

def transform_data_ing(winl,winp,trf,title=None,Total_l=None,Total_p=None):
    legend=create_legend_conprel(winl)
    winl.columns=winl.columns.map(legend)
    winp.columns=winp.columns.map(legend)
    ylabel='€'
    if trf==1:
        ylabel='%Total'
    elif trf==2:
        ylabel='€/habitante'

    if (type(Total_l)!=type(None)) and (type(Total_p)!=type(None)):
        for col in winp.columns:
            winp.loc[:,col]=winp[col]/Total_p
        for col in winl.columns:
            winl.loc[:,col]=winl[col]/Total_l



    cols=winp.sum().sort_values()
    unused_cols=cols[cols<1]
    cols=cols[cols>1]
    if cols.shape[0]<1:
        print('Non hai datos')
        return
    #frames=4
    #while len(cols)/frames>3:
    #    frames+=2

    nrows=int(np.ceil(len(cols)/3)) if len(cols)>3 else 1
    dt=int(round(len(cols)/nrows))
    figure=plt.figure(figsize=(20,6*nrows),layout='tight')

    for i in range(1,nrows+1):
        plt.subplot(nrows,2,2*i-1)
        sel_cols=cols.index[(i-1)*dt:i*dt]
        if cols.shape[0]==7 and i==nrows:

            sel_cols=cols.index[(i-1)*dt:]
        df=winl[sel_cols]
        plot_serie_conprel(df,labely=ylabel)
        plt.title('Liquidación')
        plt.subplot(nrows,2,2*i)
        df=winp[sel_cols]
        plot_serie_conprel(df,labely=ylabel)
        plt.title('Orzamento')
    plt.savefig(f'{title}.png')
    return sorted(legend.items()),unused_cols,figure

ingresos=[ 'Impuestos directos', 'Impuestos Indirectos',
       'Tasas, precios públicos y otros ingresos', 'Transferencias corrientes',
       'Ingresos patrimoniales', 'Enajenación inversiones reales',
       'Transferencias de Capital', 'Activos financieros',
       'Pasivos financieros', ]
gastos1=[ 'Gastos de personal',
       'Gastos corrientes en bienes y servicios', 'Gastos financieros',
       'Transferencias corrientes.1', 'Fondo de contingencia',
       'Inversiones reales', 'Transferencias de Capital.1',
       'Activos financieros.1', 'Pasivos financieros.1',]
gastos2=[ 'Deuda Pública', 'Servicios públicos básicos',
       'Actuaciones de protección y promoción social',
       'Producción de bienes públicos de carácter preferente',
       'Actuaciones de carácter económico', 'Actuaciones de carácter general']

# CESEL routines
def plot_serie_cesel(df,labelx=None,labely=None):
    graph=sns.lineplot(data=df,x='Ano',y='coste_efectivo',hue='Descripción')
    ymax=np.ceil(df.coste_efectivo.max())
    ymin=np.floor(df.coste_efectivo.min())
    for elec in list(range(2011,df.Ano.max()+1,4)):
        if elec>=df.Ano.min():
            plt.vlines(elec,ymin,ymax,color='red',linestyle=':')
    if labelx:
        plt.xlabel(labelx)
    if labely:
        plt.ylabel(labely)
    plt.grid()

def analisis_cesel(wa,title=None,Total=None,trf=None):
    leg_cesel=create_legend_cesel(wa)
    ylabel='€'
    if trf==1:
        ylabel='%Total'
    elif trf==2:
        ylabel='€/habitante'
    if type(Total)!=type(None):
        wa.loc[:,'coste_efectivo']= wa['coste_efectivo']/Total
    null_a=wa[wa['coste_efectivo']<0.01]
    todrop=[]
    for item in null_a.Descripción.unique():
        key=f'Descripción=="{item}"'
        vals=null_a.query(key)['Ano'].nunique()
        if vals>=wa.Ano.nunique()-1:
            todrop.append(item)

    if len(todrop)>0:
        print('Servizos NON prestados (ou só 1 ano no periodo considerado)')
        for item in todrop:
            print('    *',item)
            wa=wa[wa.Descripción != item]


    leenda_wa=[]
    for item in sorted(wa.Descripción.unique()):
        leenda_wa.append((leg_cesel[item],item))

    ng=wa.Descripción.nunique()
    nrows=int(np.ceil(ng/6))
    indx=np.ones(nrows*2)
    while ng-sum(indx)>indx.shape[0]:
        indx+=1
    total=int(ng-sum(indx))
    indx[:(total)]=indx[:(total)]+1


    wa.loc[:,'Descripción']=wa.Descripción.map(leg_cesel)

    cols=wa.groupby('Descripción')['coste_efectivo'].mean().sort_values()
    cols=list(cols.index)


    numfig=1
    figure=plt.figure(figsize=(18,5*nrows),layout='tight')

    for i,j in zip([0]+list(np.cumsum(indx)[:-1]),list(np.cumsum(indx))):
        key=(' or '.join([f'Descripción == "{item}"' for item in cols[int(i):int(j)]]))
        plt.subplot(nrows,2,numfig)
        plot_serie_cesel(wa.query(key),labely=ylabel)
        numfig+=1
    return leg_cesel,todrop,figure


def create_legend_cesel(df):
    legend={}
    for col in df['Descripción'].unique():
        sigla=''.join([item[0].upper() for item in col.split() if len(item)>3])

        index=1
        while sigla in legend.values():
            sigla+=col.split()[-1][index]
            index+=1
        legend[col]=sigla
    return legend

conceptos_cesel={'directa_publica_directos_gastos_personal':'DPD_GP',
       'directa_publica_directos_gastos_corrientes_servicios':'DPD_GCS',
       'directa_publica_directos_amortizacion':'DPD_Am',
       'directa_publica_directos_arrendamiento':'DPD_Arr',
       'directa_publica_directos_transferencias':'DPD_T',
       'directa_publica_directos_otros_no_financieros':'DPD_ONF',
       'directa_publica_indirectos':'DPI',
       'directa_empresarial_aprovisionamiento':'DE_Ap',
       'directa_empresarial_gastos_personal':'DE_GP',
       'directa_empresarial_otros_explotacion':'DE_OE',
       'directa_empresarial_amortizacion_inmovilizado':'DE_AI',
       'directa_empresarial_otros_no_financieros':'DE_ONF',
       'indirecta_contraprestaciones':'I_Ctrp'}

create_serie= lambda x: pd.DataFrame([wliq[x],wpres[x]],index=['Liquidación','Orzamento']).T

# Load Data

path_conprel=Path.cwd()/'CONPREL'
path_cesel=Path.cwd()/'CESEL'

conprel_liq=pickle_var(path_conprel/'CONPREL_Liq_all.pkl')
conprel_pres=pickle_var(path_conprel/'CONPREL_Pres_all.pkl')

cesel_a=pickle_var(path_cesel/'CESEL_all_A.pkl')
cesel_b=pickle_var(path_cesel/'CESEL_all_B.pkl')
dot_3a=pickle_var(path_cesel/'CESEL_all_3a.pkl')
dot_3b=pickle_var(path_cesel/'CESEL_all_3b.pkl')

ente=list(conprel_pres.Nombre.unique())
entidades={}
for key in cesel_a['Nombre Ente Principal'].unique():
    entidades[key]=list(cesel_a[cesel_a['Nombre Ente Principal']==key]['Nombre Ente'].unique())

not_services=set(ente)-set(entidades)


values=[]
for item in entidades.values():
    values+=item
not_services=not_services-set(values)

transformacions=['Ningunha','% Total','Por habitante']




st.title('Historico Entidades Locais Galegas')
st.markdown('Datos de liquidacións e orzamentos de [CONPREL](https://serviciostelematicosext.hacienda.gob.es/SGFAL/CONPREL), para o periodo 2010-2021 (liquidacións) e 2010-2022 (orzamentos)')
st.markdown('Datos de coste de servizos de [CESEL](https://serviciostelematicosext.hacienda.gob.es/sgcief/Cesel/Consulta/mapa/ConsultaMapa.aspx), para o periodo 2014-2021')
st.markdown('Pode atoparse información mais detallada dos orzamentos en [Información de los Presupuestos de las Entidades Locales](https://www.hacienda.gob.es/es-ES/CDI/Paginas/InformacionPresupuestaria/InformacionCCLLs/Presupuestos_EELL.aspx)')
trf=st.selectbox('transformación',transformacions)
trf=transformacions.index(trf)
print(trf)
st.header('Selecciona Entidade Local')
concello=st.selectbox('Selecciona Entidade Local ',sorted(ente),label_visibility='hidden')
print(concello)
if concello in not_services:
    st.text(f'{concello} sen servizos declarados')
elif len(entidades[concello])>1:
    conc2=st.selectbox('Selecciona entidade dependente: ',sorted(entidades[concello]),sorted(entidades[concello]).index(concello))

else:
    conc2=concello

# Proceso datos

wliq=conprel_liq[conprel_liq.Nombre==concello]
wliq=wliq.set_index('Ano',drop=True)
wliq=wliq.sort_index()
wpres=conprel_pres[conprel_pres.Nombre==concello]
wpres=wpres.set_index('Ano',drop=True)
wpres=wpres.sort_index()
total_liq=wliq[[item for item in wliq.columns if 'Total' in item]]
total_pres=wpres[[item for item in wpres.columns if 'Total' in item]]
csla=cesel_a[(cesel_a['Nombre Ente Principal']==concello) & (cesel_a['Nombre Ente']==conc2)]
total_cesel_a=pd.DataFrame(csla.groupby('Ano')['coste_efectivo'].sum())
total_cesel_a.columns=['Gasto total']
cslb=cesel_b[(cesel_b['Nombre Ente Principal']==concello) & (cesel_b['Nombre Ente']==conc2)]
total_cesel_b=pd.DataFrame(cslb.groupby('Ano')['coste_efectivo'].sum())
total_cesel_b.columns=['Gasto total']


tpl=tpo=tsa=tsb=None
if trf==2:
    tpl=wliq[['Población']].values.flatten()
    tpo=wpres[['Población']].values.flatten()
    mapping=wpres[['Población']].to_dict()['Población']
    tsa=csla['Ano'].map(mapping).values
    tsb=cslb['Ano'].map(mapping).values
elif trf==1:
    mapping=(csla.groupby('Ano')['coste_efectivo'].sum()/100).to_dict()
    tsa=csla['Ano'].map(mapping).values
    mapping=(cslb.groupby('Ano')['coste_efectivo'].sum()/100).to_dict()
    tsb=cslb['Ano'].map(mapping).values



resultados=st.button('Resultados')
if resultados:
    st.subheader(f'Descrición global')
    figure=plt.figure(figsize=(15,16),layout='tight')
    #plt.suptitle(concello)
    plt.subplot(3,2,1)
    plot_serie_conprel(wpres[['Población']],labely='Habitantes')


    plt.subplot(3,2,2)
    plot_serie_conprel(pd.DataFrame((wliq['Total ingresos']-wliq['Total gastos']).T,columns=['Diferenza Ingreso - Gasto']),labely='€')
    plt.hlines(0,wpres.index.min(),wpres.index.max(),color='red')

    plt.subplot(3,2,3)
    plot_serie_conprel(create_serie('Total ingresos'),labely='€')
    plt.title(' Ingresos ')

    plt.subplot(3,2,4)
    plot_serie_conprel(create_serie('Total gastos'),labely='€')
    plt.title(' Gastos')

    plt.subplot(3,2,5)
    plot_serie_conprel(total_cesel_a,labely='€')
    plt.title(' Servizos (A)')

    plt.subplot(3,2,6)
    plot_serie_conprel(total_cesel_b,labely='€')
    plt.title(' Servizos (B)')

    #plt.savefig('resumo.png')
    #st.image('resumo.png',width=1200)
    st.pyplot(figure,use_container_width=False)

    if trf==1:
        tpl=total_liq.iloc[:,0]/100
        tpo=total_pres.iloc[:,0]/100
    leenda,unused,figure=transform_data_ing(wliq[ingresos],wpres[ingresos],trf,title='INGRESOS',Total_l=tpl,Total_p=tpo)

    st.subheader('Ingresos')
    if len(unused):
        adnel={val:key for key,val in leenda}
        st.write(f'Conceptos non usados: {", ".join([adnel[item] for item in sorted(unused.keys())])}')
    st.write('Lenda:')
    lenda=''
    for key,val in leenda:
        lenda+=(f'\n\t{val}:\t{key}')
    st.text(lenda)
    #st.image('INGRESOS.png',width=1200)
    st.pyplot(figure,use_container_width=False)
    if trf==1:
        tpl=total_liq.iloc[:,2]/100
        tpo=total_pres.iloc[:,2]/100

    leenda,unused,figure=transform_data_ing(wliq[gastos1],wpres[gastos1],trf,title='GASTOS(A)',Total_l=tpl,Total_p=tpo)
    st.subheader('Gastos (A)')
    if len(unused):
        adnel={val:key for key,val in leenda}
        st.write(f'Conceptos non usados: {", ".join([adnel[item] for item in sorted(unused.keys())])}')
    st.write('Lenda:')
    lenda=''
    for key,val in leenda:
        lenda+=(f'\n\t{val}:\t{key}')
    st.text(lenda)
    #st.image('GASTOS(A).png',width=1200)
    st.pyplot(figure,use_container_width=False)

    leenda,unused,figure=transform_data_ing(wliq[gastos2],wpres[gastos2],trf,title='GASTOS(B)',Total_l=tpl,Total_p=tpo)
    st.subheader('Gastos (B)')
    if len(unused):
        adnel={val:key for key,val in leenda}
        st.write(f'Conceptos non usados: {", ".join([adnel[item] for item in sorted(unused.keys())])}')
    st.write('Lenda:')
    lenda=''
    for key,val in leenda:
        lenda+=(f'\n\t{val}:\t{key}')
    st.text(lenda)
    #st.image('GASTOS(B).png',width=1200)
    st.pyplot(figure,use_container_width=False)


    st.header(f'Coste efectivo dos servizos prestados por {conc2}')
    st.subheader('Grupo A')
    lenda,non_prestados,figure=analisis_cesel(csla, 'SERVIZOS (A)',tsa,trf)
    if len(lenda)<1:
        st.text(f'{conc2} non presta servizos')
    else:
        if non_prestados:

            texto='Servizos NON prestados (ou só 1 ano no periodo considerado):'
            for item in non_prestados:
                texto+=f'<br>  &nbsp;  + {item}'
            st.markdown(texto,unsafe_allow_html=True)
        #st.write('Lenda:')
        texto='Lenda: '
        for key,val in sorted(lenda.items()):
            if not key in non_prestados:
                texto+=f'<br>  &nbsp;&nbsp;&nbsp;&nbsp;  * {val}:&nbsp;&nbsp; {key}'
        st.markdown(texto,unsafe_allow_html=True)
        st.pyplot(figure,use_container_width=False)
    st.subheader('Grupo B')
    lenda,non_prestados,figure=analisis_cesel(cslb, 'SERVIZOS (B)',tsb,trf)
    if len(lenda)<1:
        st.text(f'{conc2} non presta servizos')
    else:
        if non_prestados:

            texto='Servizos NON prestados (ou só 1 ano no periodo considerado):'
            for item in non_prestados:
                texto+=f'<br>  &nbsp;  + {item}'
            st.markdown(texto,unsafe_allow_html=True)
        #st.write('Lenda:')
        texto='Lenda: '
        for key,val in sorted(lenda.items()):
            if not key in non_prestados:
                texto+=f'<br>  &nbsp;&nbsp;&nbsp;&nbsp;  * {val}:&nbsp;&nbsp; {key}'
        st.markdown(texto,unsafe_allow_html=True)
        st.pyplot(figure,use_container_width=False)
