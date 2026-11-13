"""Recuperação local no servidor + geração opcional pela Groq.

Não há fallback para outro provedor ou upgrade de plano. O plano Free precisa
ser mantido na conta Groq; não é possível inferir o plano pela chave de API.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

from rag_olist import SISTEMA, AUDITOR, FORMATO, FORMATO_AUDITOR, validar_resposta, conferir_auditoria, detalhes_acrescentados


class ErroServico(Exception):
    pass


def carregar_dados(pasta):
    import numpy as np
    info = json.loads((pasta / 'manifesto.json').read_text(encoding='utf-8'))
    for nome, hash_esperado in info['sha256'].items():
        if Path(nome).name != nome:
            raise ValueError('Nome de arquivo inválido no manifesto.')
        if hashlib.sha256((pasta / nome).read_bytes()).hexdigest() != hash_esperado:
            raise ValueError(f'Arquivo inconsistente: {nome}. Envie novamente os dados do pacote.')
    documentos = []
    for nome in info['arquivos_documentos']:
        documentos.extend(json.loads(linha) for linha in (pasta / nome).read_text(encoding='utf-8').splitlines() if linha.strip())
    partes, mapas = [], []
    for nome in info['arquivos_vetores']:
        with np.load(pasta / nome, allow_pickle=False) as a:
            partes.append(a['vetores'])
            mapas.append(a['documentos'])
    vetores, mapa = np.concatenate(partes), np.concatenate(mapas)
    if len(documentos) != info['documentos'] or vetores.shape != (info['trechos'], 384):
        raise ValueError('Contagem de dados diferente do manifesto.')
    if len(mapa) != len(vetores) or mapa.min() < 0 or mapa.max() >= len(documentos):
        raise ValueError('Mapa de avaliações inválido.')
    if not np.isfinite(vetores).all() or len({d['document_id'] for d in documentos}) != len(documentos):
        raise ValueError('Dados inválidos no índice.')
    return info, documentos, vetores, mapa


class Busca:
    def __init__(self, pasta):
        self.info, self.docs, self.vetores, self.mapa = carregar_dados(pasta)
        self.modelo = None
        self.trava = threading.Lock()

    def buscar(self, pergunta, nota_maxima=None, uf=None, k=5):
        import numpy as np
        if not isinstance(pergunta, str) or not 3 <= len(pergunta.strip()) <= 600:
            raise ValueError('A pergunta precisa ter entre 3 e 600 caracteres.')
        if type(k) is not int or not 1 <= k <= 5:
            raise ValueError('Recupere entre 1 e 5 avaliações.')
        if nota_maxima is not None and (type(nota_maxima) is not int or not 1 <= nota_maxima <= 5):
            raise ValueError('Nota máxima inválida.')
        with self.trava:
            if self.modelo is None:
                from sentence_transformers import SentenceTransformer
                cache = os.environ.get('OLIST_MODEL_CACHE', str(Path(tempfile.gettempdir()) / 'olist_modelos'))
                self.modelo = SentenceTransformer(self.info['modelo_embeddings'], cache_folder=cache,
                                                  device='cpu', trust_remote_code=False)
            q = self.modelo.encode([pergunta.strip()], normalize_embeddings=True, convert_to_numpy=True)[0]
        if not np.isfinite(q).all():
            raise ValueError('Não foi possível representar a pergunta.')
        notas = np.einsum('ij,j->i', self.vetores, q, optimize=False)
        doc_notas = np.full(len(self.docs), -np.inf, dtype='float32')
        np.maximum.at(doc_notas, self.mapa, notas)
        candidatos = []
        for i, doc in enumerate(self.docs):
            meta = doc['metadata']
            if nota_maxima is not None and (meta.get('review_score') is None or meta['review_score'] > nota_maxima):
                continue
            if uf and meta.get('customer_state') != uf:
                continue
            candidatos.append(i)
        indices = sorted(candidatos, key=lambda i: float(doc_notas[i]), reverse=True)[:k]
        return [{'posicao': p, 'similaridade': float(doc_notas[i]), **self.docs[i]} for p, i in enumerate(indices, 1)]


def chamar_groq(chave, modelo, mensagens, schema):
    # A troca de modelos/provedores nunca é automática.
    if modelo not in ('openai/gpt-oss-20b', 'openai/gpt-oss-120b'):
        raise ValueError('Modelo não habilitado neste protótipo. Use openai/gpt-oss-20b.')
    schema = copy.deepcopy(schema)
    def simplificar(obj):
        if isinstance(obj, dict):
            obj.pop('maxItems', None)
            for v in obj.values(): simplificar(v)
        elif isinstance(obj, list):
            for v in obj: simplificar(v)
    simplificar(schema)
    payload = {'model': modelo, 'messages': mensagens, 'temperature': 0,
        'max_completion_tokens': 2000, 'reasoning_effort': 'low',
        'response_format': {'type': 'json_schema', 'json_schema': {
            'name': 'olist_evidencias', 'strict': True, 'schema': schema}}}
    req = urllib.request.Request('https://api.groq.com/openai/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer '+chave})
    try:
        with urllib.request.urlopen(req, timeout=70) as r:
            dados = json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise ErroServico('A cota ou o limite temporário da Groq foi atingido. Aguarde e tente depois. Nenhum outro serviço será usado.') from None
        if exc.code in (401, 403):
            raise ErroServico('A Groq não autorizou a chave. Confira os Secrets e a conta no plano Free.') from None
        raise ErroServico(f'O serviço de geração retornou HTTP {exc.code}. As avaliações continuam disponíveis abaixo.') from None
    except (OSError, ValueError) as exc:
        raise ErroServico('O serviço de geração está indisponível. As avaliações recuperadas continuam disponíveis.') from None
    try:
        resposta = dados['choices'][0]
        if resposta.get('finish_reason') != 'stop':
            raise ValueError('Geração incompleta.')
        return json.loads(resposta['message']['content']), dados.get('usage', {})
    except (ValueError, KeyError, IndexError, TypeError):
        raise ErroServico('O serviço não retornou uma resposta completa no formato esperado.') from None


def responder(pergunta, fontes, chave, modelo='openai/gpt-oss-20b'):
    if not fontes:
        return {'suficiente': False, 'motivo': 'Nenhuma avaliação atende aos filtros.', 'achados': []}, {'chamadas': 0}
    if not chave:
        raise ErroServico('Chave Groq ausente. A busca funciona sem API; configure a geração nos Secrets.')
    contexto = [{'document_id': f['document_id'], 'texto': f['text'],
                 'nota': f['metadata']['review_score']} for f in fontes]
    inicial, uso1 = chamar_groq(chave, modelo, [{'role': 'system', 'content': SISTEMA},
        {'role': 'user', 'content': json.dumps({'pergunta': pergunta, 'avaliacoes': contexto}, ensure_ascii=False)}], FORMATO)
    # IDs/trechos desconhecidos são descartados, sem publicar a afirmação.
    if not isinstance(inicial, dict) or type(inicial.get('suficiente')) is not bool:
        raise ErroServico('Formato de resposta inválido.')
    if not inicial['suficiente']:
        validar_resposta(inicial, fontes)
        return {'suficiente': False, 'motivo': 'O modelo não identificou suporte suficiente nas avaliações recuperadas. Inspecione as fontes.', 'achados': []}, {'chamadas': 1, 'geracao': uso1}
    if not isinstance(inicial.get('achados'), list) or not 1 <= len(inicial['achados']) <= 4:
        raise ErroServico('O modelo não retornou entre 1 e 4 achados.')
    candidatos = []
    for a in inicial['achados']:
        try:
            validar_resposta({'suficiente': True, 'motivo': '', 'achados': [a]}, fontes)
            if detalhes_acrescentados(a['afirmacao'], a['trecho']):
                a = {**a, 'afirmacao': f'Um cliente relata: “{a["trecho"]}”'}
            candidatos.append(a)
        except ValueError:
            continue
    if not candidatos:
        return {'suficiente': False, 'motivo': 'As citações geradas não passaram na validação.', 'achados': []}, {'chamadas': 1, 'geracao': uso1}
    auditoria, uso2 = chamar_groq(chave, modelo, [{'role': 'system', 'content': AUDITOR},
        {'role': 'user', 'content': json.dumps({'pergunta': pergunta,
            'candidatos': [{'indice': i, **a} for i, a in enumerate(candidatos)]}, ensure_ascii=False)}], FORMATO_AUDITOR)
    aprovados = conferir_auditoria(auditoria, len(candidatos))
    achados = [a for i, a in enumerate(candidatos) if i in aprovados]
    resposta = {'suficiente': bool(achados), 'motivo': '' if achados else 'A revisão automática não aprovou os achados. Isso não demonstra ausência de relatos na base.', 'achados': achados}
    validar_resposta(resposta, fontes)
    return resposta, {'chamadas': 2, 'geracao': uso1, 'auditoria': uso2}
