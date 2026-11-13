#!/usr/bin/env python3
"""Busca semântica local: indexar uma vez, depois buscar perguntas.

python busca_olist.py indexar
python busca_olist.py buscar --pergunta "Relatos de atraso na entrega" --nota-maxima 2
Não gera respostas com LLM e não estima frequência de problemas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
MODELO = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def hash_arquivo(caminho):
    h = hashlib.sha256()
    with caminho.open('rb') as f:
        for bloco in iter(lambda: f.read(1024 * 1024), b''):
            h.update(bloco)
    return h.hexdigest()


def carregar_modelo(pasta, nome):
    # Downloads e cache ficam nesta pasta do projeto. Nenhum comentário é enviado.
    os.environ.setdefault('HF_HOME', str(pasta / 'cache_modelos'))
    from sentence_transformers import SentenceTransformer
    print('Carregando modelo local (a primeira execução precisa baixar o modelo)...', flush=True)
    return SentenceTransformer(nome, cache_folder=str(pasta / 'cache_modelos'), device='cpu', trust_remote_code=False)


def indexar(args):
    import numpy as np
    if not args.corpus.is_file():
        raise ValueError(f'Corpus não encontrado: {args.corpus}. Execute analise_olist.py primeiro.')
    documentos = [json.loads(linha) for linha in args.corpus.read_text(encoding='utf-8').splitlines() if linha.strip()]
    ids = [d['document_id'] for d in documentos]
    if not documentos or len(set(ids)) != len(ids):
        raise ValueError('O corpus está vazio ou tem document_id repetido.')
    if any(not d.get('text', '').strip() for d in documentos):
        raise ValueError('O corpus contém texto vazio.')
    assinatura = hash_arquivo(args.corpus)
    modelo = carregar_modelo(args.indice, args.modelo)
    # Divide por tokens para não descartar silenciosamente o fim de textos longos.
    tok = modelo.tokenizer
    tamanho = modelo.max_seq_length - tok.num_special_tokens_to_add(pair=False)
    if tamanho <= 0:
        raise ValueError('Limite de tokens inválido no modelo.')
    passo = max(1, tamanho - min(20, tamanho // 4))
    trechos, mapa = [], []
    for i, doc in enumerate(documentos):
        tokens = tok.encode(doc['text'], add_special_tokens=False)
        for inicio in range(0, len(tokens), passo):
            trecho = tok.decode(tokens[inicio:inicio+tamanho], skip_special_tokens=True)
            trechos.append(trecho)
            mapa.append(i)
            if inicio + tamanho >= len(tokens):
                break
    print(f'Criando embeddings de {len(trechos):,} trechos de {len(documentos):,} avaliações. Aguarde a barra terminar.', flush=True)
    vetores = modelo.encode(trechos, batch_size=32, show_progress_bar=True,
                            convert_to_numpy=True, normalize_embeddings=True)
    if hash_arquivo(args.corpus) != assinatura:
        raise ValueError('Corpus alterado durante a indexação. Execute novamente.')
    args.indice.mkdir(parents=True, exist_ok=True)
    # NPZ sem pickle: vetores normalizados + índice da avaliação de origem.
    np.savez_compressed(args.indice / 'vetores.npz', vetores=vetores.astype('float32'),
                        documentos=np.asarray(mapa, dtype='int64'))
    snapshot = args.indice / 'documentos.json'
    snapshot.write_text(json.dumps(documentos, ensure_ascii=False), encoding='utf-8')
    info = {'modelo': args.modelo, 'corpus_sha256': assinatura,
            'documentos': len(documentos), 'trechos': len(trechos),
            'documentos_sha256': hash_arquivo(snapshot),
            'vetores_sha256': hash_arquivo(args.indice / 'vetores.npz'),
            'max_seq_length': modelo.max_seq_length, 'sobreposicao_tokens': tamanho-passo}
    (args.indice / 'config.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    print(f'Índice concluído: {args.indice}')


def recuperar(args):
    """Recupera avaliações para uso na busca e no pipeline RAG."""
    import numpy as np
    config = args.indice / 'config.json'
    if not config.is_file():
        raise ValueError('Índice ausente. Execute: python busca_olist.py indexar')
    info = json.loads(config.read_text(encoding='utf-8'))
    for nome, chave in [('vetores.npz', 'vetores_sha256'), ('documentos.json', 'documentos_sha256')]:
        if hash_arquivo(args.indice / nome) != info[chave]:
            raise ValueError('O índice está inconsistente. Execute indexar novamente.')
    if args.corpus.is_file() and hash_arquivo(args.corpus) != info['corpus_sha256']:
        raise ValueError('O corpus mudou. Execute indexar novamente para atualizar a busca.')
    documentos = json.loads((args.indice / 'documentos.json').read_text(encoding='utf-8'))
    with np.load(args.indice / 'vetores.npz', allow_pickle=False) as arquivo:
        vetores = arquivo['vetores']
        mapa = arquivo['documentos']
    modelo = carregar_modelo(args.indice, info['modelo'])
    query = modelo.encode([args.pergunta], convert_to_numpy=True, normalize_embeddings=True)[0]
    if not np.isfinite(vetores).all() or not np.isfinite(query).all():
        raise ValueError('Embeddings contêm valores inválidos. Confira ou recrie o índice.')
    # Evita avisos numéricos de algumas combinações NumPy/BLAS no macOS.
    similaridades = np.einsum('ij,j->i', vetores, query, optimize=False)
    if not np.isfinite(similaridades).all():
        raise ValueError('A comparação retornou similaridades inválidas.')
    # Vários trechos da mesma avaliação não podem ocupar várias posições no top-k.
    notas_doc = np.full(len(documentos), -np.inf, dtype='float32')
    np.maximum.at(notas_doc, mapa, similaridades)
    candidatos = []
    for i, doc in enumerate(documentos):
        nota = doc['metadata'].get('review_score')
        if args.nota_maxima is not None and (nota is None or nota > args.nota_maxima):
            continue
        if args.uf and doc['metadata'].get('customer_state') != args.uf.upper():
            continue
        candidatos.append(i)
    melhores = sorted(candidatos, key=lambda i: float(notas_doc[i]), reverse=True)[:args.top_k]
    saida = []
    for posicao, i in enumerate(melhores, 1):
        doc = documentos[i]
        item = {'posicao': posicao, 'similaridade': float(notas_doc[i]), **doc}
        saida.append(item)
    return saida


def buscar(args):
    saida = recuperar(args)
    print('\nAvaliações recuperadas (similaridade não é probabilidade nem prova de relevância):')
    for item in saida:
        doc = item
        meta = doc['metadata']
        print(f'\n{item["posicao"]}. [{doc["document_id"]}] similaridade={item["similaridade"]:.3f} | nota={meta.get("review_score")} | UF={meta.get("customer_state")}')
        print(doc['text'])
    if not saida:
        print('Nenhuma avaliação atende aos filtros.')
    print('\nInspecione os textos: a busca sempre ordena candidatos, inclusive para perguntas fora do escopo.')
    print('Esta etapa ainda não decide se há evidência suficiente e não gera uma resposta RAG.')
    destino = args.indice / 'ultima_busca.json'
    destino.write_text(json.dumps({'pergunta': args.pergunta, 'filtros': {'nota_maxima': args.nota_maxima, 'uf': args.uf},
                                  'resultados': saida}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Resultados salvos em: {destino}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=RAIZ / 'resultados/documentos_rag.jsonl')
    parser.add_argument('--indice', type=Path, default=RAIZ / 'resultados/indice_semantico')
    subs = parser.add_subparsers(dest='comando', required=True)
    idx = subs.add_parser('indexar', help='Criar ou substituir o índice local')
    idx.add_argument('--modelo', default=MODELO)
    busca = subs.add_parser('buscar', help='Recuperar avaliações para uma pergunta')
    busca.add_argument('--pergunta', required=True)
    busca.add_argument('--top-k', type=int, default=5)
    busca.add_argument('--nota-maxima', type=int, choices=range(1, 6))
    busca.add_argument('--uf', help='Filtrar pela UF do cliente, por exemplo SP')
    args = parser.parse_args()
    if args.comando == 'buscar' and (not args.pergunta.strip() or args.top_k < 1):
        parser.error('Informe uma pergunta e --top-k maior que zero.')
    try:
        (indexar if args.comando == 'indexar' else buscar)(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(1, f'Erro: {exc}\n')


if __name__ == '__main__':
    main()
