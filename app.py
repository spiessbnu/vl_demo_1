import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging

# --- Configuração da Página e Logger ---
st.set_page_config(
    page_title="Tutor de Matemática",
    page_icon="🤖",
    layout="centered"
)

# Configuração de um logger simples para nosso "Terminal"
# Em vez de prints, usamos um logger para ter mais controle no futuro.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Função para adicionar logs ao nosso terminal na UI
def log_to_terminal(message):
    st.session_state.log_messages.append(message)
    logging.info(message)

# --- Carregamento de Dados (com Cache) ---
@st.cache_data
def carregar_dados():
    """Carrega o DataFrame e extrai a matriz de embeddings.
    Usa o cache do Streamlit para executar esta função apenas uma vez."""
    log_to_terminal("Iniciando carregamento dos dados...")
    try:
        df = pd.read_parquet("dados_curriculares_enriquecidos.parquet")
        matriz_embeddings = np.array(df['embedding'].tolist())
        log_to_terminal("Dados carregados com sucesso!")
        return df, matriz_embeddings
    except FileNotFoundError:
        st.error("Arquivo 'dados_curriculares_enriquecidos.parquet' não encontrado. Verifique o caminho.")
        log_to_terminal("ERRO: Arquivo de dados não encontrado.")
        return None, None

# --- Funções do Núcleo de IA (RAG) ---
def gerar_embedding_query(texto, client):
    """Gera o embedding para a query do usuário."""
    log_to_terminal(f"Gerando embedding para a query: '{texto[:30]}...'")
    try:
        response = client.embeddings.create(input=[texto], model="text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        log_to_terminal(f"ERRO na API de Embeddings: {e}")
        return None

def buscar_conteudo_relevante(query_embedding, df, matriz_embeddings, ano_aluno, top_k=5):
    """Busca, calcula similaridade e ranqueia o conteúdo."""
    if query_embedding is None:
        return pd.DataFrame()

    log_to_terminal("Calculando similaridade de cosseno...")
    # Calcula a similaridade entre a query e todos os itens do currículo
    scores = cosine_similarity([query_embedding], matriz_embeddings)[0]

    # Adiciona os scores ao DataFrame para facilitar a manipulação
    df['similaridade'] = scores

    log_to_terminal(f"Ranqueando resultados. Prioridade para o {ano_aluno}º ano.")
    # Ordena os resultados: primeiro critério é se o ano é o do aluno, segundo é a similaridade
    df_ranqueado = df.sort_values(
        by=['Ano', 'similaridade'],
        ascending=[True, False],
        key=lambda col: col if col.name != 'Ano' else col != ano_aluno
    )

    resultados = df_ranqueado.head(top_k)
    log_to_terminal("Top 5 resultados brutos (Índice | Ano | Score):")
    for i, row in resultados.iterrows():
        log_to_terminal(f"- {i} | {row['Ano']}º ano | {row['similaridade']:.4f}")

    return resultados.iloc[[0]] # Retorna apenas o melhor resultado

# --- Inicialização da Aplicação ---
df, matriz_embeddings = carregar_dados()

# Verifica se a chave da API foi configurada nos segredos do Streamlit
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Chave da API da OpenAI não encontrada. Por favor, configure o arquivo .streamlit/secrets.toml")
    st.stop()

# Inicializa o estado da sessão se não existir
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "aluno_ano" not in st.session_state:
    st.session_state.aluno_ano = None

# --- Interface do Usuário (UI) ---

st.title("🤖 Tutor Inteligente de Matemática")
st.caption("Um assistente baseado no currículo de SC para te ajudar a estudar.")

# Sidebar para configuração e debug
with st.sidebar:
    st.header("Configurações")
    anos_disponiveis = sorted(df['Ano'].unique()) if df is not None else []
    
    # Seleção do ano do aluno
    ano_selecionado = st.selectbox(
        "Qual ano você está cursando?",
        options=anos_disponiveis,
        index=4 # Padrão para o 5º ano, por exemplo
    )
    st.session_state.aluno_ano = ano_selecionado

    st.divider()

    # Terminal de Debug
    with st.expander("🔌 Terminal de Debug", expanded=True):
        log_container = st.container(height=300)
        log_text = "\n".join(st.session_state.log_messages)
        log_container.text(log_text)
        
# Exibe o histórico do chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Captura a entrada do usuário
if prompt := st.chat_input("O que vamos estudar hoje?"):
    # Adiciona a mensagem do usuário ao histórico e à tela
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Inicia o processo de resposta do assistente
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            # Limpa logs antigos para a nova query
            st.session_state.log_messages = []
            log_to_terminal("--- NOVA QUERY RECEBIDA ---")

            # Passo 1: Gerar embedding para a query do usuário
            query_embedding = gerar_embedding_query(prompt, client)

            # Passo 2: Buscar e ranquear o conteúdo relevante (RAG - Retrieve)
            df_contexto = buscar_conteudo_relevante(
                query_embedding, df, matriz_embeddings, st.session_state.aluno_ano
            )

            # Prepara a mensagem de contexto para o LLM
            if not df_contexto.empty:
                contexto_curricular = df_contexto.iloc[0]['texto_completo']
                fonte = f"Fonte: Currículo do {df_contexto.iloc[0]['Ano']}º ano - {df_contexto.iloc[0]['Objetos do conhecimento']}"
                log_to_terminal("Contexto selecionado para a API.")
                
                # Constrói a mensagem do sistema para o LLM
                system_prompt = f"""
                Você é um tutor de matemática amigável, paciente e didático.
                Sua missão é ajudar um aluno do {st.session_state.aluno_ano}º ano.
                Use o seguinte CONTEXTO CURRICULAR para basear sua resposta. Não invente informações.
                Seja claro, use exemplos simples e sempre responda em português do Brasil.

                CONTEXTO CURRICULAR:
                {contexto_curricular}
                """

                # Junta o histórico para dar memória ao chatbot
                mensagens_para_api = [{"role": "system", "content": system_prompt}]
                for msg in st.session_state.messages:
                    mensagens_para_api.append(msg)

                log_to_terminal("Enviando requisição para a API 'gpt-4o-mini'...")
                
                # Passo 3: Gerar a resposta com o LLM (RAG - Generate)
                try:
                    stream = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=mensagens_para_api,
                        stream=True,
                    )
                    # Usa st.write_stream para um efeito de "digitação"
                    response = st.write_stream(stream)
                    
                    # Adiciona a resposta completa ao histórico
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    log_to_terminal("Resposta da API recebida e exibida.")

                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
                    log_to_terminal(f"ERRO na API de Chat: {e}")

            else:
                st.write("Não consegui encontrar um conteúdo diretamente relacionado no currículo. Você pode tentar reformular a pergunta?")
                log_to_terminal("Nenhum contexto relevante encontrado.")
    
    # Força o rerender da página para atualizar o terminal de debug
    st.rerun()
