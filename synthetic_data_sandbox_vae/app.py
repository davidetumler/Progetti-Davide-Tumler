"""Dashboard Streamlit: avviare tramite avvia_mac.sh o streamlit run app.py."""
from __future__ import annotations

from io import BytesIO
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API = 'http://127.0.0.1:8000'
st.set_page_config(page_title='Synthetic Data Sandbox', page_icon='🧬', layout='wide')
st.title('🧬 Synthetic Data Sandbox — VAE Edition')
st.caption('Progetto Master | PyTorch · FastAPI · Streamlit · Analisi in linguaggio naturale')
st.warning('Demo educativa locale: i dati sintetici non sono automaticamente anonimi. '
           'Non caricare dati personali reali senza adeguate garanzie e autorizzazioni.')


def request(method: str, path: str, **kwargs):
    try:
        response = requests.request(method, API + path, timeout=600, **kwargs)
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        detail = str(exc)
        if getattr(exc, 'response', None) is not None:
            try:
                detail = exc.response.json().get('detail', detail)
            except ValueError:
                pass
        st.error(f'Errore API: {detail}')
        return None


def load_preview(source: str, rows: int = 100) -> pd.DataFrame | None:
    response = request('GET', '/preview', params={'source': source, 'rows': rows})
    if response is None:
        return None
    payload = response.json()
    return pd.read_json(BytesIO(payload['records'].encode('utf-8')), orient='records')


status_response = request('GET', '/status')
if status_response is None:
    st.info('Avvia prima il backend FastAPI con bash avvia_mac.sh.')
    st.stop()
status = status_response.json()

tab1, tab2, tab3, tab4 = st.tabs(['1. Carica CSV', '2. Addestra VAE', '3. Genera e valida', '4. AI Analyst'])

with tab1:
    st.subheader('Carica un dataset tabellare')
    upload = st.file_uploader('Seleziona un CSV (massimo 10 MB)', type=['csv'])
    st.caption('Per una prova usa example_data.csv, incluso nel progetto.')
    if st.button('Carica nel backend', disabled=upload is None):
        response = request('POST', '/upload', files={'file': (upload.name, upload.getvalue(), 'text/csv')})
        if response is not None:
            st.success(f"Caricati {response.json()['rows']} record e {len(response.json()['columns'])} colonne")
            st.rerun()
    if status['uploaded']:
        st.dataframe(load_preview('real', rows=10), use_container_width=True)

with tab2:
    st.subheader('Addestramento Variational Autoencoder')
    epochs = st.slider('Epoche', 5, 100, 25, step=5)
    latent_dim = st.slider('Dimensioni spazio latente', 2, 16, 4)
    if st.button('Avvia addestramento', disabled=not status['uploaded']):
        with st.spinner('Addestramento in corso...'):
            response = request('POST', '/train', json={'epochs': epochs, 'latent_dim': latent_dim})
        if response is not None:
            st.success('Modello addestrato!')
            st.line_chart(pd.DataFrame({'Loss': response.json()['loss']}))
            st.rerun()
    if status['trained']:
        st.info('Il VAE è addestrato. Vai nella scheda Genera e valida.')

with tab3:
    st.subheader('Genera dati sintetici')
    rows = st.number_input('Numero di record da generare', min_value=20, max_value=10000, value=300, step=20)
    if st.button('Genera nuovi record', disabled=not status['trained']):
        response = request('POST', '/generate', json={'rows': int(rows)})
        if response is not None:
            st.success(f"Generati {response.json()['rows']} record sintetici")
            st.rerun()
    if status['generated']:
        real, fake = load_preview('real'), load_preview('synthetic')
        left, right = st.columns(2)
        with left:
            st.markdown('**Dati originali (anteprima)**')
            st.dataframe(real.head(10), use_container_width=True)
        with right:
            st.markdown('**Dati sintetici (anteprima)**')
            st.dataframe(fake.head(10), use_container_width=True)
        csv_response = request('GET', '/download')
        if csv_response is not None:
            st.download_button('Scarica CSV sintetico', data=csv_response.content,
                               file_name='synthetic_data.csv', mime='text/csv')
        st.markdown('#### Confronto distribuzioni')
        numeric = real.select_dtypes(include='number').columns.tolist()
        if numeric:
            col = st.selectbox('Variabile da confrontare', numeric)
            chart_data = pd.concat([
                pd.DataFrame({'Valore': real[col], 'Origine': 'Reale'}),
                pd.DataFrame({'Valore': fake[col], 'Origine': 'Sintetico'})], ignore_index=True)
            st.plotly_chart(px.histogram(chart_data, x='Valore', color='Origine',
                                        barmode='overlay', nbins=30), use_container_width=True)
        if len(numeric) > 1:
            a, b = st.columns(2)
            a.plotly_chart(px.imshow(real[numeric].corr(), text_auto='.2f',
                                     title='Correlazioni reali', zmin=-1, zmax=1), use_container_width=True)
            b.plotly_chart(px.imshow(fake[numeric].corr(), text_auto='.2f',
                                     title='Correlazioni sintetiche', zmin=-1, zmax=1), use_container_width=True)
        metrics = request('GET', '/validation')
        if metrics is not None:
            v = metrics.json()
            st.metric('Corrispondenze esatte (arrotondate)', v['exact_matches_rounded'])
            st.metric('Errore medio assoluto correlazioni',
                      v['correlation_mean_absolute_error'] if v['correlation_mean_absolute_error'] is not None else 'N/D')
            st.caption(v['warning'])

with tab4:
    st.subheader('Natural Language Analyst')
    st.write('Le domande vengono applicate **esclusivamente ai dati sintetici**.')
    question = st.text_input('Scrivi una domanda',
                             'Qual è la correlazione tra reddito e risparmio?')
    if st.button('Analizza', disabled=not status['generated']):
        response = request('POST', '/analyze', json={'question': question})
        if response is not None:
            result = response.json()
            st.success(result['answer'])
            fake = load_preview('synthetic')
            if fake is not None:
                kind, cols = result['kind'], result['columns']
                if kind == 'correlation':
                    fig = px.scatter(fake, x=cols[0], y=cols[1],
                                     trendline=None, title='Relazione (anteprima sintetica)')
                    st.plotly_chart(fig, use_container_width=True)
                elif kind == 'histogram':
                    st.plotly_chart(px.histogram(fake, x=cols[0], nbins=25), use_container_width=True)
                elif kind == 'heatmap':
                    st.plotly_chart(px.imshow(fake[cols].corr(), text_auto='.2f', zmin=-1, zmax=1),
                                    use_container_width=True)
                elif kind == 'average':
                    st.metric(f'Media: {cols[0]}', result['value'])
    st.caption('Con OPENAI_API_KEY l’AI interpreta le intenzioni. Senza chiave sono disponibili comandi base. '
               'Nessun codice Python generato dall’AI viene eseguito.')
