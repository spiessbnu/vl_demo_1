import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
import re

# --- Configuração da Página e Logger ---
st.set_page_config(page_title="Tutor de Matemática", page_icon="🤖", layout="centered")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# --- INICIALIZAÇÃO DO SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "aluno_ano" not in st.session_state:
    st.session_state.aluno_ano = 5

def log_to_terminal(message):
    st.session_state.log_messages.append(message)
    logging.info(message)

# --- FUNÇÃO DE QUERY CONTEXTUALIZADA (NOVA) ---
def criar_query_contextualizada(historico_mensagens: list, max_turnos=2) -> str:
    """
    Cria uma string de busca rica em contexto combinando as últimas interações.
    """
    log_to_terminal("Criando query contextualizada para a busca...")
    # Pega as últimas N mensagens (max_turnos * 2 = mensagens de user e assistant)
    mensagens_relevantes = historico_mensagens[-max_turnos*2:]
    
    query_contextualizada = " ".join(
        [f"{msg['role']}: {msg['content']}" for msg in mensagens_relevantes]
    )
    return query_contextualizada


def corrigir_latex_inline(texto: str) -> str:
    pattern = r'(?<!\$)\\(frac|rac)\{([^\}]+)\}\{([^\}]+)\}(?!\$)'
    def normalizar_e_delimitar(match):
        numerador = match.group(2)
        denominador = match.group(3)
        return f"$\\frac{{{numerador}}}{{{denominador}}}$"
    return re.sub(pattern, normalizar_e_delimitar, texto)

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

try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Chave da API da OpenAI não encontrada. Configure o arquivo .streamlit/secrets.toml")
    st.stop()

df, matriz_embeddings = carregar_dados()
if df is None: st.stop()

st.title("🤖 Tutor Inteligente de Matemática")
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

def renderizar_mensagem(message):
    if message["role"] == "user":
        st.markdown(message["content"])
        return
    try:
        data = json.loads(message["content"])
        for block in data.get("response", []):
            block_type = block.get("type")
            content = block.get("content")
            if block_type == "paragraph":
                st.markdown(corrigir_latex_inline(content))
            elif block_type == "math_block":
                st.latex(content)
            elif block_type == "list":
                list_md = ""
                for item in block.get("items", []):
                    list_md += f"- {corrigir_latex_inline(item)}\n"
                st.markdown(list_md)
    except (json.JSONDecodeError, TypeError):
        st.markdown(corrigir_latex_inline(message["content"]))

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        renderizar_mensagem(message)

if prompt := st.chat_input("O que vamos estudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            st.session_state.log_messages = []
            log_to_terminal("--- NOVA QUERY RECEBIDA ---")
            
            # --- MUDANÇA PRINCIPAL AQUI ---
            # 1. Cria a query contextualizada a partir do histórico
            query_para_rag = criar_query_contextualizada(st.session_state.messages)
            log_to_terminal(f"Query Contextualizada para RAG: '{query_para_rag}'")

            # 2. Usa a nova query para a busca semântica
            query_embedding = gerar_embedding_query(query_para_rag, client)
            # ---------------------------------

            df_contexto = buscar_conteudo_relevante(
                query_embedding, df.copy(), matriz_embeddings, st.session_state.aluno_ano
            )
            if not df_contexto.empty:
                contexto_row = df_contexto.iloc[0]
                contexto_curricular = contexto_row['texto_completo']

                log_to_terminal("\n--- CONTEXTO SELECIONADO PARA O LLM ---")
                log_to_terminal(f"Índice: {contexto_row.name}, Ano: {contexto_row['Ano']}, Score: {contexto_row['similaridade']:.4f}")
                log_to_terminal("---------------------------------------\n" + contexto_curricular + "\n---------------------------------------\n")

                # O system_prompt e a chamada ao LLM continuam iguais,
                # pois já passávamos o histórico completo para a GERAÇÃO da resposta.
                # A mudança foi na RECUPERAÇÃO do contexto.
                system_prompt = f"""
                Você é um tutor de matemática. Sua resposta DEVE ser um objeto JSON válido.
                A estrutura do JSON é uma lista de blocos de conteúdo chamada "response".
                Tipos de blocos: "paragraph", "math_block", "list".
                - "paragraph": para texto, pode conter LaTeX inline com um cifrão ($).
                - "math_block": APENAS o código LaTeX, sem cifrões.
                - "list": um array de strings chamado "items".
                REGRAS DE FORMATAÇÃO:
                1. Blocos matemáticos ($$) devem estar em seu próprio bloco "math_block".
                2. Fórmulas inline ($) devem estar dentro de um "paragraph" ou "list".
                3. Use espaços antes e depois de blocos matemáticos.
                4. Para números mistos, inclua o inteiro DENTRO dos cifrões, ex: `$1\\frac{{1}}{{4}}$`.

                CONTEXTO CURRICULAR: {contexto_curricular}
                """
                # Para o LLM, ainda enviamos o histórico separado para ele entender o fluxo da conversa
                mensagens_para_api = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                
                log_to_terminal("Enviando requisição para API (modo JSON)...")
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini", messages=mensagens_para_api,
                        response_format={"type": "json_object"}
                    )
                    resposta_json_str = response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": resposta_json_str})
                    log_to_terminal("Resposta JSON da API recebida.")
                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
                    log_to_terminal(f"ERRO na API de Chat: {e}")
            else:
                fallback_msg = {"response": [{"type": "paragraph", "content": "Não consegui encontrar um conteúdo diretamente relacionado no currículo. Você pode tentar reformular a pergunta?"}]}
                st.session_state.messages.append({"role": "assistant", "content": json.dumps(fallback_msg)})
                log_to_terminal("Nenhum contexto relevante encontrado.")
    st.rerun()
