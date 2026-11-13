# Tech Challenge - Fase 4

Olist · Voz do Cliente no Streamlit

Aplicação educacional para buscar avaliações da Olist e, opcionalmente, gerar
respostas citando evidências. A busca funciona sem chave de API. A geração usa
Groq, com conta no plano Free e sujeita às cotas desse plano.

## O que está pronto

- Interface Streamlit com filtros de nota e UF e download da consulta em JSON.
- Índice existente dividido em arquivos menores, mantendo os vetores float32.
- Validação de IDs e citações, guardas pontuais de detalhes adicionais e auditoria por IA.
- Até duas chamadas à Groq por consulta, sem repetição automática ou troca de provedor.
- Tratamento de limite de cota, serviço indisponível e chave ausente.
- Senha opcional para compartilhar o protótipo com um grupo.

Esta pasta é uma versão separada da aplicação local com Ollama. A mudança de
modelo exige nova avaliação humana. As métricas obtidas com Ollama não se
transferem automaticamente para o modelo da Groq.

## Publicar, passo a passo

1. No GitHub, crie um repositório chamado `olist-voz-cliente`. Para não expor
   o código e os comentários em um repositório público, você pode escolher
   **Private** e autorizar o Streamlit a acessar esse repositório.
2. Envie o **conteúdo desta pasta** para a raiz do repositório. O arquivo
   `streamlit_app.py` deve aparecer ao lado de `requirements.txt` e da pasta
   `dados/`. Envie os arquivos extraídos, não o ZIP. As partes do índice estão
   abaixo de 24 MiB cada. Não envie a pasta `.venv`, logs ou chaves.
3. Entre em [Streamlit Community Cloud](https://share.streamlit.io/) com sua
   conta GitHub. Clique em **Create app** e selecione seu repositório, branch e
   o arquivo `streamlit_app.py`. Em **Advanced settings**, escolha Python 3.12.
4. Você já pode publicar o modo de busca sem API. Para respostas geradas,
   crie uma chave em uma conta no plano **Free** da [Groq](https://console.groq.com/).
   Mantenha a conta Free e não habilite Developer nem outro plano pago.
5. Em **Advanced settings → Secrets**, ou **Settings → Secrets** depois da
   publicação, configure:

   ```toml
   GROQ_API_KEY = "SUA_CHAVE_REAL_APENAS_AQUI"
   GROQ_MODEL = "openai/gpt-oss-20b"
   # Opcional: para restringir consultas ao seu grupo
   APP_PASSWORD = "UMA_SENHA_ESCOLHIDA_POR_VOCE"
   ```

6. Clique em **Deploy**. No primeiro uso, aguarde o download do modelo de
   embeddings. A aplicação abre em um endereço `*.streamlit.app`.
7. Teste o modo de busca. Para testar o RAG, marque **Gerar resposta com IA**
   na barra lateral e faça uma pergunta. Confira as citações e faça nova revisão
   com perguntas variadas antes de apresentar a solução como validada.

A chave deve ficar nos Secrets, nunca no GitHub, no chat ou em capturas de tela.
Não existe chave real neste pacote. `secrets.example.toml` é apenas um exemplo.

## Sem cobrança: condições e limites

O [Streamlit Community Cloud é gratuito](https://docs.streamlit.io/deploy/streamlit-community-cloud),
mas tem recursos limitados e pode suspender ou reduzir a disponibilidade da aplicação.
A [Groq distingue o plano Free do Developer pago](https://console.groq.com/docs/billing-faqs)
e aplica [limites de uso por organização e modelo](https://console.groq.com/docs/rate-limits).

Este código não verifica o plano da conta pela chave. A configuração precisa
usar uma conta Free para atender ao requisito de não ter cobrança. Não há
upgrade, assinatura nem fallback para serviço pago implementados no código.
Se a cota terminar, a geração informa o limite; as fontes recuperadas continuam
visíveis. Há um intervalo de 30 segundos entre inícios de geração neste processo.
As cotas reais são aplicadas pela Groq e podem mudar. Não há promessa de uso ilimitado.

## Dados e arquitetura

O corpus deriva dos nove CSVs Olist fornecidos para o Tech Challenge.
`dados/manifesto.json` registra o modelo, a contagem e os hashes de cada parte.
A busca usa o mesmo `paraphrase-multilingual-MiniLM-L12-v2` e os mesmos vetores
do índice local. O arquivo de documentos inclui texto, ID da avaliação, registro
de origem e metadados reduzidos, sem IDs de cliente ou coordenadas.

Fluxo: pergunta → embedding no servidor Streamlit → comparação com vetores →
até cinco avaliações → geração opcional na Groq → validação de IDs e trechos →
auditoria adicional na Groq → resposta e fontes. O esquema estruturado segue a
[documentação da Groq](https://console.groq.com/docs/structured-outputs).

Perguntas e comentários recuperados são enviados à Groq quando a geração é
ativada. No modo de busca, esse envio não ocorre. O modelo de embeddings é
baixado do Hugging Face e executado no servidor. A conta Free da Groq precisa
ser configurada pelo proprietário da aplicação.

## Rodar localmente

Na pasta desta versão, com o ambiente Python ativado:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py --server.port 8502
```

A porta 8502 evita conflito com a aplicação local anterior em 8501. Sem Secrets
o modo de busca funciona. Para configurar a geração localmente, copie o exemplo
para `.streamlit/secrets.toml` e preencha a chave ali. Esse arquivo é ignorado
pelo Git e excluído do pacote preparado.

## Testes e limitações

```bash
python -m unittest teste_cloud.py
```

Os testes usam respostas simuladas para conferir integração, citações e limites.
Não comprovam precisão do modelo nem disponibilidade na conta Groq. Sem uma
chave configurada, não foi realizado teste de geração real pela API.

A revisão automática pode aceitar erros ou rejeitar fontes úteis. As regras
lexicais cobrem apenas exemplos observados, sem garantia semântica geral.
Top-k não estima frequência na base inteira. Categorias são do pedido e não
identificam necessariamente o item sobre o qual o cliente escreveu.
Recursos e memória devem ser conferidos na implantação real do Community Cloud.

No servidor, os recursos de busca são compartilhados em cache com trava na
codificação de perguntas. Resultados ficam separados por sessão. O armazenamento
do Streamlit Cloud não deve ser tratado como persistência: use o download JSON.
