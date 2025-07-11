import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
import re

# --- Configuração da Página e Logger ---
st.set_page_config(page_title="VL demo 1 - Matemática", page_icon="🤖", layout="centered")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- INICIALIZAÇÃO DO SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "aluno_ano" not in st.session_state:
    st.session_state.aluno_ano = 5

def log_to_terminal(message):
    st.session_state.log_messages.append(str(message))
    logging.info(message)

def corrigir_latex_inline(texto: str) -> str:
    pattern = r'(?<!\$)\\(frac|rac)\{([^\}]+)\}\{([^\}]+)\}(?!\$)'
    def normalizar_e_delimitar(match):
        numerador = match.group(2)
        denominador = match.group(3)
        return f"$\\frac{{{numerador}}}{{{denominador}}}$"
    return re.sub(pattern, normalizar_e_delimitar, texto)

# --- Funções de Carregamento de Dados e RAG (sem alterações) ---
@st.cache_data
def carregar_dados():
    log_to_terminal("Iniciando carregamento dos dados...")
    try:
        df = pd.read_parquet("dados_curriculares_enriquecidos.parquet")
        matriz_embeddings = np.array(df['embedding'].tolist())
        return df, matriz_embeddings
    except FileNotFoundError:
        st.error("Arquivo 'dados_curriculares_enriquecidos.parquet' não encontrado.")
        return None, None

def gerar_embedding_query(texto, client):
    log_to_terminal(f"Gerando embedding para a query contextualizada...")
    try:
        response = client.embeddings.create(input=[texto], model="text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None

def buscar_conteudo_relevante(query_embedding, df, matriz_embeddings, ano_aluno, top_k=5):
    if query_embedding is None: return pd.DataFrame()
    log_to_terminal("Calculando similaridade de cosseno...")
    scores = cosine_similarity([query_embedding], matriz_embeddings)[0]
    df['similaridade'] = scores
    log_to_terminal(f"Ranqueando resultados. Prioridade para o {ano_aluno}º ano.")
    df_ranqueado = df.sort_values(
        by=['Ano', 'similaridade'], ascending=[True, False],
        key=lambda col: col if col.name != 'Ano' else col != ano_aluno
    )
    resultados = df_ranqueado.head(top_k)
    log_to_terminal("Top 5 resultados (Índice | Ano | Score):")
    for i, row in resultados.iterrows():
        log_to_terminal(f"- {i} | {row['Ano']}º ano | {row['similaridade']:.4f}")
    return resultados.iloc[[0]]

def criar_query_contextualizada(historico_mensagens: list, max_turnos=2) -> str:
    log_to_terminal("Criando query contextualizada para a busca...")
    mensagens_relevantes = historico_mensagens[-max_turnos*2:]
    query_contextualizada = " ".join(
        [f"{msg['role']}: {msg['content']}" for msg in mensagens_relevantes]
    )
    return query_contextualizada

# --- Inicialização e UI ---
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Chave da API da OpenAI não encontrada. Configure o arquivo .streamlit/secrets.toml")
    st.stop()

df, matriz_embeddings = carregar_dados()
if df is None: st.stop()

st.title("🤖 VL demo 1 - Matemática")
st.caption("Um assistente baseado no currículo de SC para te ajudar a estudar.")

with st.sidebar:
    st.header("Configurações")
    anos_disponiveis = sorted(df['Ano'].unique())
    ano_selecionado = st.selectbox(
        "Qual ano você está cursando?", options=anos_disponiveis,
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

# --- NOVA LÓGICA DE RENDERIZAÇÃO SIMPLIFICADA ---
def renderizar_mensagem(message):
    if message["role"] == "user":
        st.markdown(message["content"])
        return

    try:
        data = json.loads(message["content"])
        # O valor de 'response' agora é uma simples lista de strings
        for line in data.get("response", []):
            # Aplicamos a correção e renderizamos cada linha
            st.markdown(corrigir_latex_inline(line))
    except (json.JSONDecodeError, TypeError):
        # Fallback para o caso de a resposta não ser um JSON
        st.markdown(corrigir_latex_inline(message["content"]))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        renderizar_mensagem(message)

# --- LÓGICA PRINCIPAL DO CHAT ---
if prompt := st.chat_input("O que vamos estudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            st.session_state.log_messages = []
            log_to_terminal("--- NOVA QUERY RECEBIDA ---")
            query_para_rag = criar_query_contextualizada(st.session_state.messages)
            log_to_terminal(f"Query Contextualizada para RAG: '{query_para_rag}'")
            query_embedding = gerar_embedding_query(query_para_rag, client)
            df_contexto = buscar_conteudo_relevante(
                query_embedding, df.copy(), matriz_embeddings, st.session_state.aluno_ano
            )
            if not df_contexto.empty:
                contexto_row = df_contexto.iloc[0]
                contexto_curricular = contexto_row['texto_completo']
                log_to_terminal("\n--- CONTEXTO SELECIONADO PARA O LLM ---")
                log_to_terminal(f"Índice: {contexto_row.name}, Ano: {contexto_row['Ano']}, Score: {contexto_row['similaridade']:.4f}")
                log_to_terminal("---------------------------------------\n" + contexto_curricular + "\n---------------------------------------\n")
                
                # --- NOVO PROMPT DO SISTEMA SIMPLIFICADO ---
                system_prompt = f"""
                Você é um tutor de matemática. Sua resposta DEVE ser um objeto JSON válido.
                O JSON deve conter uma única chave: "response".
                O valor de "response" deve ser uma lista de strings. Cada string é um parágrafo ou um item de lista (começando com '- ').
                Use Markdown e LaTeX (com $...$) para formatar o texto dentro das strings.

                Exemplo de resposta JSON válida:
                {{
                  "response": [
                    "A fórmula de Bhaskara é usada para resolver equações de segundo grau.",
                    "A fórmula é: $$\\Delta = b^2 - 4ac$$",
                    "- Onde $a$, $b$, e $c$ são os coeficientes da equação.",
                    "- O valor de $x$ é encontrado com $x = \\frac{{-b \\pm \\sqrt{{\\Delta}}}}{{2a}}$."
                  ]
                }}

                Agora, usando o CONTEXTO CURRICULAR abaixo, responda à pergunta do aluno do {st.session_state.aluno_ano}º ano seguindo ESTRITAMENTE o formato JSON.
                CONTEXTO CURRICULAR: {contexto_curricular}
                """
                mensagens_para_api = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                
                log_to_terminal("Enviando requisição para API (modo JSON simplificado)...")
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini", messages=mensagens_para_api,
                        response_format={"type": "json_object"}
                    )
                    resposta_json_str = response.choices[0].message.content
                    log_to_terminal("\n--- RESPOSTA BRUTA DA API (JSON) ---")
                    log_to_terminal(resposta_json_str)
                    st.session_state.messages.append({"role": "assistant", "content": resposta_json_str})
                    log_to_terminal("Resposta JSON adicionada ao histórico.")
                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
                    log_to_terminal(f"ERRO na API de Chat: {e}")
            else:
                fallback_msg = {"response": ["Não consegui encontrar um conteúdo diretamente relacionado no currículo. Você pode tentar reformular a pergunta?"]}
                st.session_state.messages.append({"role": "assistant", "content": json.dumps(fallback_msg)})
                log_to_terminal("Nenhum contexto relevante encontrado.")
    st.rerun()
