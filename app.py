import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
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

def log_to_terminal(message):
    st.session_state.log_messages.append(message)
    logging.info(message)

# --- FUNÇÃO DE CORREÇÃO DE LATEX (VERSÃO 2.0 - MAIS INTELIGENTE) ---
def corrigir_notacao_latex(texto: str) -> str:
    """
    Encontra qualquer bloco de texto que contenha comandos LaTeX, remove
    quaisquer delimitadores '$' inconsistentes ao redor dele e, em seguida,
    envolve o conteúdo limpo com '$$...$$' para uma exibição em bloco correta.
    """
    # Padrão para encontrar conteúdo que parece LaTeX (contém um '\' seguido de letras)
    # e que pode ou não estar (mal) envolvido por '$' ou '$$'.
    pattern = r'\${0,2}(.*?\\(?:frac|cdot|sqrt|sum|pi|alpha|beta|[a-zA-Z]+\^\{?[0-9]\}?|times|div|neq|leq|geq).*?)\${0,2}'

    def normalizar_bloco(match):
        # Pega o conteúdo interno do bloco encontrado
        conteudo = match.group(1)
        # Remove quaisquer '$' do início ou do fim do conteúdo capturado
        conteudo_limpo = conteudo.strip().strip('$')
        # Retorna o conteúdo limpo, agora corretamente formatado como um bloco matemático
        return f"$$\n{conteudo_limpo}\n$$"

    # Usa re.sub com a função de callback para substituir todos os blocos encontrados
    texto_corrigido = re.sub(pattern, normalizar_bloco, texto)
    
    return texto_corrigido


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
        log_to_terminal("ERRO: Arquivo de dados não encontrado.")
        return None, None

def gerar_embedding_query(texto, client):
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

try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Chave da API da OpenAI não encontrada. Por favor, configure o arquivo .streamlit/secrets.toml")
    st.stop()

df, matriz_embeddings = carregar_dados()
if df is None:
    st.stop()

st.title("🤖 Tutor Inteligente de Matemática")
st.caption("Um assistente baseado no currículo de SC para te ajudar a estudar.")

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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("O que vamos estudar hoje?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            st.session_state.log_messages = []
            log_to_terminal("--- NOVA QUERY RECEBIDA ---")
            query_embedding = gerar_embedding_query(prompt, client)
            df_contexto = buscar_conteudo_relevante(
                query_embedding, df.copy(), matriz_embeddings, st.session_state.aluno_ano
            )
            if not df_contexto.empty:
                contexto_row = df_contexto.iloc[0]
                contexto_curricular = contexto_row['texto_completo']

                log_to_terminal("\n--- CONTEXTO SELECIONADO PARA O LLM ---")
                log_to_terminal(f"Índice: {contexto_row.name}, Ano: {contexto_row['Ano']}, Score: {contexto_row['similaridade']:.4f}")
                log_to_terminal("---------------------------------------\n" + contexto_curricular + "\n---------------------------------------\n")

                # --- PROMPT DO SISTEMA REFORÇADO ---
                system_prompt = f"""
                Você é um tutor de matemática amigável e didático para um aluno do {st.session_state.aluno_ano}º ano.
                Baseie sua resposta no CONTEXTO CURRICULAR fornecido.
                
                REGRAS RÍGIDAS DE FORMATAÇÃO MATEMÁTICA:
                1.  Para fórmulas em bloco ou equações importantes, use SEMPRE dois cifrões no início e no fim. Exemplo: $$\\frac{{a}}{{b}} + \\frac{{c}}{{d}} = \\frac{{ad+bc}}{{bd}}$$
                2.  Para pequenas variáveis ou frações no meio de uma frase, use um único cifrão. Exemplo: A variável $x$ é igual a $\\frac{{1}}{{2}}$.
                3.  NUNCA misture os formatos, como `$$\frac{a}{b}$` ou `$\frac{a}{b}$$`.

                CONTEXTO CURRICULAR:
                {contexto_curricular}
                """
                mensagens_para_api = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages:
                    mensagens_para_api.append(msg)
                log_to_terminal("Enviando requisição para a API 'gpt-4o-mini'...")
                try:
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=mensagens_para_api,
                        stream=True,
                    )
                    resposta_completa = ""
                    placeholder = st.empty()
                    for chunk in stream:
                        resposta_completa += (chunk.choices[0].delta.content or "")
                        placeholder.markdown(resposta_completa + "▌")
                    
                    # Usa a nova função para limpar e formatar a resposta
                    resposta_corrigida = corrigir_notacao_latex(resposta_completa)
                    placeholder.markdown(resposta_corrigida)
                    
                    st.session_state.messages.append({"role": "assistant", "content": resposta_corrigida})
                    log_to_terminal("Resposta da API recebida e exibida.")
                    if resposta_completa != resposta_corrigida:
                        log_to_terminal("Notação LaTeX corrigida na resposta da API.")
                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
                    log_to_terminal(f"ERRO na API de Chat: {e}")
            else:
                st.write("Não consegui encontrar um conteúdo diretamente relacionado no currículo. Você pode tentar reformular a pergunta?")
                log_to_terminal("Nenhum contexto relevante encontrado.")
    st.rerun()
