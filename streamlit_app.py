"""Entrada da aplicação para Streamlit Community Cloud."""
from __future__ import annotations
import hmac
import json
import os
import threading
import time
from pathlib import Path

import streamlit as st
from base_cloud import Busca, ErroServico, responder

RAIZ = Path(__file__).resolve().parent
st.set_page_config(page_title='Tech Challenge - Fase 4', page_icon='🎓', layout='wide')
st.markdown('''<style>
.stApp {background:#090a0d;color:#f4f4f7}
[data-testid="stSidebar"] {background:#121318;border-right:1px solid #2b2c34}
[data-testid="stHeader"] {background:#090a0d}
h1 {letter-spacing:-1.3px;font-weight:600!important}
.fiap-marca {color:#ed145b;font-size:30px;font-style:italic;letter-spacing:3px;font-weight:650;margin-bottom:22px}
.fiap-marca span {color:#a7a9b4;font-size:11px;letter-spacing:1.6px;font-style:normal;margin-left:20px}
.fiap-curso {color:#ff6093;font-size:11px;text-transform:uppercase;letter-spacing:2px;margin-bottom:5px}
</style>''', unsafe_allow_html=True)
st.markdown('<div class="fiap-marca">FIAP <span>PROJETO ACADÊMICO · PÓS TECH</span></div>', unsafe_allow_html=True)


def segredo(nome, padrao=''):
    try:
        return str(st.secrets.get(nome, os.environ.get(nome, padrao)))
    except FileNotFoundError:
        return os.environ.get(nome, padrao)


@st.cache_resource
def indice():
    return Busca(RAIZ / 'dados')


@st.cache_resource
def limite_global():
    # Compartilhado entre sessões deste processo: reduz rajadas à API.
    return {'trava': threading.Lock(), 'proxima': 0.0}


senha = segredo('APP_PASSWORD')
if senha and not st.session_state.get('autorizado', False):
    st.title('Tech Challenge - Fase 4')
    with st.form('acesso'):
        tentativa = st.text_input('Senha de acesso ao protótipo', type='password')
        entrou = st.form_submit_button('Entrar')
    if entrou:
        if hmac.compare_digest(tentativa.encode(), senha.encode()):
            st.session_state['autorizado'] = True
            st.rerun()
        else:
            st.error('Senha incorreta.')
    st.stop()

st.markdown('<div class="fiap-curso">Pós Tech / Inteligência Artificial</div>', unsafe_allow_html=True)
st.title('Tech Challenge - Fase 4')
st.subheader('Voice of Customer Intelligence')
st.write('Investigue as avaliações e confira as evidências por trás de cada resposta.')
st.caption('Protótipo educacional · Dados históricos da Olist · Consultas independentes')
with st.sidebar:
    st.header('Filtros')
    nota = st.selectbox('Nota máxima', ['Todas', 1, 2, 3, 4, 5])
    uf = st.selectbox('UF do cliente', ['Todas']+'AC AL AP AM BA CE DF ES GO MA MT MS MG PA PB PR PE PI RJ RN RS RO RR SC SP SE TO'.split())
    k = st.selectbox('Avaliações a recuperar', [3, 5], index=1)
    chave = segredo('GROQ_API_KEY')
    chave_valida = bool(chave and not chave.startswith('COLOQUE_'))
    gerar = st.checkbox('Gerar resposta com IA', value=False, disabled=not chave_valida)
    if not chave_valida:
        st.info('A busca funciona sem chave. Para gerar respostas, configure GROQ_API_KEY nos Secrets usando uma conta Free.')
    else:
        st.caption('Geração configurada na Groq. Mantenha a conta no plano Free; há limites de uso.')
    st.caption('Ao gerar uma resposta, sua pergunta e os comentários recuperados são enviados à Groq. A busca sem geração usa apenas o servidor do Streamlit.')
    st.caption('A aplicação não verifica o plano de cobrança da conta pela chave e não faz upgrades ou troca de provedor.')

with st.form('consulta'):
    pergunta = st.text_area('O que você quer entender?', placeholder='O que os clientes relatam sobre demora na entrega?', max_chars=600)
    enviar = st.form_submit_button('Consultar avaliações', type='primary')

if enviar:
    if not 3 <= len(pergunta.strip()) <= 600:
        st.warning('Escreva uma pergunta com pelo menos três caracteres.')
    else:
        entrada = {'pergunta': pergunta.strip(), 'nota_maxima': None if nota == 'Todas' else nota,
                   'uf': None if uf == 'Todas' else uf, 'top_k': k}
        resultado = {'consulta': entrada, 'fontes': [], 'resposta': None, 'modo': 'busca', 'erro': None}
        try:
            with st.spinner('Buscando avaliações. O primeiro acesso pode precisar baixar o modelo de embeddings…'):
                resultado['fontes'] = indice().buscar(entrada['pergunta'], entrada['nota_maxima'], entrada['uf'], k)
            if gerar:
                limite = limite_global()
                with limite['trava']:
                    agora = time.monotonic()
                    if agora < limite['proxima']:
                        raise ErroServico('A geração está em intervalo de espera. Aguarde alguns segundos; as fontes já estão disponíveis.')
                    limite['proxima'] = agora + 30
                with st.spinner('Gerando e conferindo as citações…'):
                    resposta, uso = responder(entrada['pergunta'], resultado['fontes'], chave, segredo('GROQ_MODEL', 'openai/gpt-oss-20b'))
                    resultado.update(resposta=resposta, uso=uso, modo='rag_groq')
        except (ErroServico, ValueError, OSError) as exc:
            resultado['erro'] = str(exc)
        except Exception:
            resultado['erro'] = 'Não foi possível concluir a consulta. Confira as dependências e os dados nos logs do Streamlit.'
        st.session_state['resultado'] = resultado

resultado = st.session_state.get('resultado')
if resultado:
    st.subheader('Resultado da consulta')
    st.write(resultado['consulta']['pergunta'])
    if resultado.get('erro'):
        st.warning(resultado['erro'])
    r = resultado.get('resposta')
    if r:
        if r['suficiente']:
            for a in r['achados']:
                st.write(a['afirmacao'])
                st.caption('Fonte: '+a['document_id'])
                st.text(a['trecho'])
                st.divider()
        else:
            st.info(r['motivo'])
    elif not resultado.get('erro'):
        st.info('Modo de busca: estas são avaliações recuperadas. Nenhuma resposta foi gerada por IA.')
    st.subheader('Avaliações recuperadas')
    st.caption('Candidatos da busca; similaridade não é probabilidade de acerto. Confira as fontes mesmo quando o modelo recusar a resposta.')
    for f in resultado['fontes']:
        m = f['metadata']
        with st.expander(f'{f["document_id"]} · Nota {m.get("review_score")} · UF {m.get("customer_state")}'):
            st.text(f['text'])
            st.caption(f'Similaridade: {f["similaridade"]:.3f}')
            st.caption('Categorias do pedido: '+', '.join(m.get('categorias') or []))
    st.download_button('Baixar consulta JSON', json.dumps(resultado, ensure_ascii=False, indent=2),
                       file_name='consulta_olist.json', mime='application/json')
else:
    st.info('Comece por uma pergunta sobre entrega, produtos ou experiência de compra.')
st.caption('O protótipo pode recusar fontes úteis ou gerar afirmações imprecisas. Frequência na base inteira não pode ser estimada pelo top-k. O modelo da nuvem é diferente do Ollama e precisa de nova avaliação.')
