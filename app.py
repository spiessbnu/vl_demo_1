import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
import re

# --- Configuração da Página e Logger ---
st.set_page_config(
    page_title="Tutor de Matemática",
    page_icon="🤖",
    layout="centered"
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- INICIALIZAÇÃO DO SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "aluno_ano" not in st.session_state:
    st.session_state.aluno_ano = 5

def log_to_terminal(message: str):
    st.session_state.log_messages.append(message)
    logging.info(message)

# --- CORREÇÃO INLINE DE LaTeX ---
def corrigir_latex_inline(texto: str) -> str:
    """
    Envolve ocorrências de \frac{a}{b} com delimitadores $…$,
    desde que ainda não estejam dentro de $…$.
    """
    pattern = r'(?<!\$)(\\frac\{[^}]+\}\{[^}]+\})(?!\$)'
    return re.sub(pattern, r'$\1$', texto)

# --- Funções de Carregamento de Dados e RAG ---
@st.cache_data
def carregar_dados():
    log_to_terminal("Iniciando carregamento dos dados...")
    try:
        df = pd.read_parquet("dados_curriculares_enriquecidos.parquet")
        matriz_embeddings = np.array(df['embedding'].tolist())
        log_to_terminal("Dados carregados com sucesso!")
        return df, matriz_embeddings
    except FileNotFoundError:
        st.error("Arquivo 'dados_curriculares_enriquecidos.parquet' não encontrado.")
        return None, None

def gerar_embedding_query(texto: str, client) -> np.ndarray:
    log_to_terminal(f"Gerando embedding para a query: '{texto[:30]}...'")
    try:
        response = client.embeddings.create(input=[texto], model="text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None

def buscar_conteudo_relevante(query_embedding, df, matriz_embeddings, ano_aluno, top_k=5):
    if query_embedding is None:
        return pd.DataFrame()
    log_to_terminal("Calculando similaridade de cosseno...")
    scores = cosine_similarity([query_embedding], matriz_embeddings)[0]
    df['similaridade'] = scores
    log_to_terminal(f"Ranqueando resultados. Prioridade para o {ano_aluno}º ano.")
    df_ranqueado = df.sort_values(
        by=['Ano', 'similaridade'],
        ascending=[True, False],
        key=lambda col: col if col.name != 'Ano' else col != ano_aluno
    )
    resultados = df_ranqueado.head(top_k)
    log_to_terminal("Top 5 resultados (Índice | Ano | Score):")
    for i, row in resultados.iterrows():
        log_to_terminal(f"- {i} | {row['Ano']}º ano | {row['similaridade']:.4f}")
    return resultados.iloc[[0]]

# --- Inicialização do Cliente OpenAI ---
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Chave da API da OpenAI não encontrada. Configure o arquivo .streamlit/secrets.toml")
    st.stop()

# --- Carrega Dados ---
df, matriz_embeddings = carregar_dados()
if df is None:
    st.stop()

# --- Cabeçalho da UI ---
st.title("🤖 Tutor Inteligente de Matemática")
st.caption("Um assistente baseado no currículo de SC para te ajudar a estudar.")

# --- Sidebar ---
with st.sidebar:
    st.header("Configurações")
    anos_disponiveis = sorted(df['Ano'].unique())
    ano_selecionado = st.selectbox(
        "Qual ano você está cursando?",
        options=anos_disponiveis,
        index=anos_disponiveis.index(st.session_state.aluno_ano)
    )
    if ano_selecionado != st.session_state.aluno_ano:
        st.session_state.aluno_ano = ano_selecionado
        st.rerun()
    st.divider()
    with st.expander("🔌 Terminal de Debug", expanded=True):
        log_container = st.container(height=300)
        log_text = "\n".join(st.session_state.log_messages)
        log_container.text(log_text)

# --- Função de Renderização de Mensagens ---
def renderizar_mensagem(message):
    if message["role"] == "user":
        st.markdown(message["content"])
        return

    try:
        data = json.loads(message["content"])
        for block in data.get("response", []):
            tipo = block.get("type")
            cont = block.get("content", "")
            if tipo == "paragraph":
                st.markdown(corrigir_latex_inline(cont), unsafe_allow_html=True)
            elif tipo == "math_block":
                st.latex(cont)
            elif tipo == "list":
                list_md = "\n".join(
                    f"- {corrigir_latex_inline(item)}"
                    for item in block.get("items", [])
                )
                st.markdown(list_md, unsafe_allow_html=True)
    except (json.JSONDecodeError, TypeError):
        st.markdown(message["content"], unsafe_allow_html=True)

# --- Exibe Mensagens Anteriores ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        renderizar_mensagem(message)

# --- Lógica Principal do Chat ---
if prompt := st.chat_input("O que vamos estudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            st.session_state.log_messages = []
            log_to_terminal("--- NOVA QUERY RECEBIDA ---")

            # Gera embedding e busca contexto
            query_embedding = gerar_embedding_query(prompt, client)
            df_contexto = buscar_conteudo_relevante(
                query_embedding,
                df.copy(),
                matriz_embeddings,
                st.session_state.aluno_ano
            )

            if not df_contexto.empty:
                contexto_row = df_contexto.iloc[0]
                contexto_curricular = contexto_row['texto_completo']

                log_to_terminal("\n--- CONTEXTO SELECIONADO PARA O LLM ---")
                log_to_terminal(
                    f"Índice: {contexto_row.name}, Ano: {contexto_row['Ano']}, "
                    f"Score: {contexto_row['similaridade']:.4f}"
                )
                log_to_terminal("---------------------------------------")
                log_to_terminal(contexto_curricular)
                log_to_terminal("---------------------------------------\n")

                # Prompt do sistema como raw f-string
                system_prompt = fr"""
Você é um tutor de matemática. Sua resposta DEVE ser um objeto JSON válido.
A estrutura do JSON é uma lista de blocos de conteúdo chamada "response".

Tipos de blocos disponíveis:
- "paragraph": texto explicativo. Pode conter LaTeX inline com cifrão, ex: $x=1$.
- "math_block": equações importantes. Conteúdo é APENAS o código LaTeX, sem cifrões.
- "list": listas. Conteúdo é um array de strings chamado "items".

Exemplo de resposta JSON válida:
{{
  "response": [
    {{ "type": "paragraph", "content": "Para somar as frações $\frac{{1}}{{2}}$ e $\frac{{1}}{{3}}$, primeiro encontramos o MMC." }},
    {{ "type": "math_block", "content": "\frac{{1}}{{2}} + \frac{{1}}{{3}} = \frac{{3+2}}{{6}} = \frac{{5}}{{6}}" }},
    {{ "type": "list", "items": ["O numerador é 5.", "O denominador é 6."] }}
  ]
}}

Agora, usando o CONTEXTO CURRICULAR abaixo, responda à pergunta do aluno do {st.session_state.aluno_ano}º ano seguindo ESTRITAMENTE o formato JSON.
CONTEXTO CURRICULAR: {contexto_curricular}
"""
                mensagens_para_api = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]

                log_to_terminal("Enviando requisição para API (modo JSON)...")
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=mensagens_para_api,
                        response_format={"type": "json_object"}
                    )
                    resposta_json_str = response.choices[0].message.content
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": resposta_json_str
                    })
                    log_to_terminal("Resposta JSON da API recebida.")
                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
                    log_to_terminal(f"ERRO na API de Chat: {e}")
            else:
                fallback = {
                    "response": [
                        {
                            "type": "paragraph",
                            "content": "Não consegui encontrar conteúdo relacionado no currículo. Tente reformular a pergunta."
                        }
                    ]
                }
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": json.dumps(fallback)
                })
                log_to_terminal("Nenhum contexto relevante encontrado.")

    st.rerun()
