# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import openai
from sklearn.metrics.pairwise import cosine_similarity
import logging
import json
import re

# --- 1. CONFIGURAÇÃO INICIAL E SESSION STATE ---
# --- ALTERADO: Título da página ---
st.set_page_config(page_title="Vibe Learning - Teste", page_icon="🤖", layout="centered")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

if "app_state" not in st.session_state:
    st.session_state.app_state = "COLETA_INFO"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []
if "aluno_ano" not in st.session_state:
    st.session_state.aluno_ano = None
if "aluno_nome" not in st.session_state:
    st.session_state.aluno_nome = ""
if "df_com_similaridade" not in st.session_state:
    st.session_state.df_com_similaridade = None
if "topico_selecionado_idx" not in st.session_state:
    st.session_state.topico_selecionado_idx = None
if "initial_action_taken" not in st.session_state:
    st.session_state.initial_action_taken = False
if 'sugestao_pendente' not in st.session_state:
    st.session_state.sugestao_pendente = None
if 'analise_feita_para_pergunta' not in st.session_state:
    st.session_state.analise_feita_para_pergunta = 0
if "desempenho_status" not in st.session_state:
    st.session_state.desempenho_status = None
if "topicos_superados" not in st.session_state:
    st.session_state.topicos_superados = []
if "mostrando_lista_relacionados" not in st.session_state:
    st.session_state.mostrando_lista_relacionados = False

# --- 2. DEFINIÇÃO DE TODAS AS FUNÇÕES AUXILIARES ---
def log_to_terminal(message):
    st.session_state.log_messages.append(str(message))
    logging.info(message)

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
    log_to_terminal(f"Gerando embedding para a query: '{texto}'")
    try:
        response = client.embeddings.create(input=[texto], model="text-embedding-3-large")
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Erro ao gerar embedding: {e}")
        return None

def calcular_similaridades_iniciais(query, df, matriz_embeddings, client):
    if not query: return None
    log_to_terminal("Calculando similaridades para a busca inicial...")
    embedding = gerar_embedding_query(query, client)
    if embedding is None: return None
    scores = cosine_similarity([embedding], matriz_embeddings)[0]
    df_com_scores = df.copy()
    df_com_scores['similaridade'] = scores
    log_to_terminal("Cálculo de similaridade inicial concluído.")
    return df_com_scores

def extrair_dados_iniciais(texto_usuario, client):
    log_to_terminal("Extraindo dados da primeira mensagem do usuário...")
    prompt_extracao = f"""
    Extraia o nome do aluno, o ano (como um número inteiro) e o assunto principal da frase abaixo.
    Responda APENAS com um objeto JSON com as chaves "nome", "ano" e "assunto".
    Frase: "{texto_usuario}"
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_extracao}],
            response_format={"type": "json_object"}
        )
        dados = json.loads(response.choices[0].message.content)
        log_to_terminal(f"Dados extraídos: {dados}")
        return dados
    except Exception as e:
        log_to_terminal(f"Erro ao extrair dados iniciais: {e}")
        return None

def renderizar_mensagem(message):
    texto_bruto = message["content"]
    texto_processado = re.sub(r'\\\[(.*?)\\\]', r'$$\1$$', texto_bruto, flags=re.DOTALL)
    texto_processado = re.sub(r'\\\((.*?)\\\)', r'$\1$', texto_processado, flags=re.DOTALL)
    st.markdown(texto_processado, unsafe_allow_html=True)

def analisar_progresso_do_topico(historico, topico_atual_idx, df_curriculo, client, topicos_superados):
    log_to_terminal("Iniciando análise de progresso pedagógico...")
    topico_atual = df_curriculo.loc[topico_atual_idx]
    
    df_potenciais = df_curriculo[
        (df_curriculo['Ano'] == topico_atual['Ano']) &
        (df_curriculo['Unidade Temática'] == topico_atual['Unidade Temática']) &
        (df_curriculo.index != topico_atual_idx) &
        (~df_curriculo.index.isin(topicos_superados))
    ]
    proximos_topicos_potenciais_nomes = df_potenciais['Objetos do conhecimento'].tolist()

    topicos_superados_nomes = df_curriculo.loc[topicos_superados]['Objetos do conhecimento'].tolist()
    
    historico_texto = "\n".join([f"{m['role']}: {m['content']}" for m in historico[-6:]])

    prompt_pedagogico = f"""
    Você é um assistente pedagógico especialista em matemática. Sua tarefa é analisar uma conversa entre um tutor e um aluno para decidir o próximo passo.

    CONTEXTO ATUAL:
    - Tópico: "{topico_atual['Objetos do conhecimento']}"
    - Habilidades esperadas: "{topico_atual['Habilidades']}"
    - Ano: {topico_atual['Ano']}º
    - Tópicos já superados nesta sessão: {topicos_superados_nomes}

    HISTÓRICO RECENTE DA CONVERSA:
    {historico_texto}

    POTENCIAIS PRÓXIMOS TÓPICOS (NÃO SUGIRA OS JÁ SUPERADOS):
    {proximos_topicos_potenciais_nomes}

    ANÁLISE:
    Com base na conversa, o aluno demonstrou compreensão do tópico atual? Ele parece estar pronto para avançar, ou está com dificuldades em um pré-requisito?

    RESPONDA APENAS COM UM OBJETO JSON com as seguintes chaves:
    - "analise_pedagogica": "Um resumo curto (1 frase) da sua análise."
    - "acao_sugerida": Escolha uma destas opções: "continuar", "avancar" ou "revisar".
    - "proximo_topico_sugerido": Se a ação for "avancar", indique qual dos 'POTENCIAIS PRÓXIMOS TÓPICOS' é a melhor sugestão. Caso contrário, deixe nulo.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt_pedagogico}],
            response_format={"type": "json_object"}
        )
        dados = json.loads(response.choices[0].message.content)
        dados['topicos_relacionados'] = proximos_topicos_potenciais_nomes
        log_to_terminal(f"Análise pedagógica recebida: {dados}")
        return dados
    except Exception as e:
        log_to_terminal(f"Erro na análise pedagógica: {e}")
        return None

# --- 3. INICIALIZAÇÃO DE OBJETOS GLOBAIS ---
try:
    client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
except (KeyError, FileNotFoundError):
    st.error("Chave da API da OpenAI não encontrada. Configure o arquivo .streamlit/secrets.toml")
    st.stop()

df, matriz_embeddings = carregar_dados()
if df is None: st.stop()


# --- 4. LÓGICA PRINCIPAL DA APLICAÇÃO (MÁQUINA DE ESTADOS) ---
# --- ALTERADO: Título principal ---
st.title("Vibe Learning - Teste")

# ESTADO 1: COLETA DE INFORMAÇÕES
if st.session_state.app_state == "COLETA_INFO":
    st.info("👋 Olá! Para começarmos, diga seu nome, o ano que você está cursando e o assunto que gostaria de estudar hoje.")
    st.caption("Exemplo: 'Meu nome é Ana, sou do 8º ano e quero aprender sobre o teorema de Pitágoras.'")

    if prompt := st.chat_input("Diga seu nome, ano e assunto..."):
        dados_iniciais = extrair_dados_iniciais(prompt, client)
        if dados_iniciais:
            st.session_state.aluno_nome = dados_iniciais.get("nome", "estudante")
            st.session_state.aluno_ano = dados_iniciais.get("ano", 8)
            assunto = dados_iniciais.get("assunto", prompt)
            df_com_similaridade = calcular_similaridades_iniciais(assunto, df, matriz_embeddings, client)
            if df_com_similaridade is not None:
                st.session_state.df_com_similaridade = df_com_similaridade
                st.session_state.app_state = "SELECAO_TOPICO"
                st.rerun()

# ESTADO 2: SELEÇÃO DE TÓPICO
elif st.session_state.app_state == "SELECAO_TOPICO":
    st.markdown(f"### Olá, {st.session_state.aluno_nome}!")
    st.markdown(f"Com base no que você pediu, encontrei estes tópicos do **{st.session_state.aluno_ano}º ano**, ordenados por relevância. Qual deles você gostaria de estudar?")
    
    df_filtrado = st.session_state.df_com_similaridade
    playlist_df = df_filtrado[df_filtrado["Ano"] == st.session_state.aluno_ano].sort_values("similaridade", ascending=False)
    
    top_hit_idx = playlist_df.index[0] if not playlist_df.empty else None

    if top_hit_idx is None:
        st.warning("Não encontrei nenhum tópico correspondente no currículo do seu ano. Tente uma busca diferente na tela inicial.")
        if st.button("Voltar ao início"):
            st.session_state.app_state = "COLETA_INFO"
            st.rerun()
    else:
        for idx, row in playlist_df.head(5).iterrows():
            with st.container(border=True):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"**{row['Objetos do conhecimento']}**")
                    st.caption(f"Unidade: {row['Unidade Temática']} | Similaridade: {row['similaridade']:.2f}")
                    if idx == top_hit_idx:
                        st.success("✨ Mais relevante")
                with col2:
                    if st.button("Estudar este", key=f"topic_{idx}"):
                        st.session_state.topico_selecionado_idx = idx
                        st.session_state.app_state = "CHAT"
                        st.session_state.initial_action_taken = False
                        st.session_state.messages = [
                            {"role": "assistant", "content": f"Ok, {st.session_state.aluno_nome}! Vamos focar em **{df.loc[idx, 'Objetos do conhecimento']}**. O que você gostaria de saber? Me peça uma explicação, exemplos ou exercícios!"}
                        ]
                        st.session_state.desempenho_status = "analisando"
                        st.session_state.sugestao_pendente = None
                        st.session_state.analise_feita_para_pergunta = 0
                        st.session_state.topicos_superados = []
                        st.rerun()

# ESTADO 3: CHAT
elif st.session_state.app_state == "CHAT":
    if st.button("⬅️ Mudar de Tópico"):
        st.session_state.messages = []
        st.session_state.app_state = "SELECAO_TOPICO"
        st.session_state.desempenho_status = None
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            renderizar_mensagem(message)
    
    if st.session_state.get('mostrando_lista_relacionados', False):
        with st.container(border=True):
            st.markdown("### Outros temas relacionados")
            st.write("Qual destes você gostaria de explorar agora?")
            
            topicos_para_mostrar = st.session_state.sugestao_pendente.get('topicos_relacionados', [])
            
            for topico_nome in topicos_para_mostrar:
                if st.button(topico_nome, key=f"outro_{topico_nome}", use_container_width=True):
                    st.session_state.topicos_superados.append(st.session_state.topico_selecionado_idx)
                    novo_idx = df[df['Objetos do conhecimento'] == topico_nome].index[0]
                    st.session_state.topico_selecionado_idx = novo_idx
                    st.session_state.messages.append({"role": "assistant", "content": f"Perfeito! Vamos estudar **{topico_nome}**. Por onde começamos?"})
                    
                    st.session_state.mostrando_lista_relacionados = False
                    st.session_state.sugestao_pendente = None
                    st.session_state.initial_action_taken = False
                    st.session_state.analise_feita_para_pergunta = 0
                    st.session_state.desempenho_status = 'analisando'
                    st.rerun()

    elif st.session_state.get('sugestao_pendente') and not st.session_state.get('mostrando_lista_relacionados'):
        sugestao = st.session_state.sugestao_pendente
        with st.container(border=True):
            st.markdown(f"💡 **Uma sugestão para você, {st.session_state.aluno_nome}!**")
            st.write(f"Notei que você está indo muito bem! {sugestao['analise_pedagogica']}")
            st.write(f"Você se sente à vontade para avançarmos para o tópico **{sugestao['proximo_topico_sugerido']}**?")
            
            col1, col2, col3 = st.columns(3)
            if col1.button("Sim, vamos avançar!", use_container_width=True):
                st.session_state.topicos_superados.append(st.session_state.topico_selecionado_idx)
                novo_topico_nome = sugestao['proximo_topico_sugerido']
                novo_topico_idx = df[df['Objetos do conhecimento'] == novo_topico_nome].index[0]
                st.session_state.topico_selecionado_idx = novo_topico_idx
                st.session_state.messages.append({"role": "assistant", "content": f"Excelente! Vamos começar a explorar **{novo_topico_nome}**. O que gostaria de saber?"})
                
                st.session_state.sugestao_pendente = None
                st.session_state.initial_action_taken = False
                st.session_state.analise_feita_para_pergunta = 0
                st.session_state.desempenho_status = 'analisando'
                st.rerun()
                
            if col2.button("Não, continuar aqui", use_container_width=True):
                st.session_state.sugestao_pendente = None
                st.rerun()

            if col3.button("Ver outros temas", use_container_width=True):
                st.session_state.mostrando_lista_relacionados = True
                st.rerun()
    
    prompt_gerado = None
    if not st.session_state.sugestao_pendente and not st.session_state.mostrando_lista_relacionados:
        if not st.session_state.get("initial_action_taken", False):
            st.write("Escolha uma ação:")
            col1, col2, col3 = st.columns(3)
            if col1.button("Explique o tópico", use_container_width=True): prompt_gerado = "Por favor, me dê uma explicação detalhada sobre este tópico."
            if col2.button("Me dê um exemplo", use_container_width=True): prompt_gerado = "Pode me dar um exemplo prático sobre isso?"
            if col3.button("Quero exercícios", use_container_width=True): prompt_gerado = "Gostaria de alguns exercícios para praticar."
            if prompt_gerado: st.session_state.initial_action_taken = True
        else:
            prompt_gerado = st.chat_input("Faça outra pergunta ou peça mais exercícios!")

    if prompt_gerado:
        st.session_state.messages.append({"role": "user", "content": prompt_gerado})
        with st.chat_message("user"): st.markdown(prompt_gerado)
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                topico_selecionado = df.loc[st.session_state.topico_selecionado_idx]
                system_prompt = f"Você é um tutor de matemática amigável e prestativo. Use o CONTEXTO CURRICULAR abaixo para responder a pergunta do aluno. Seja claro e use exemplos simples. Você pode usar formatação Markdown e LaTeX (com delimitadores \\(para inline\\) e \\[para bloco\\]).\n\nCONTEXTO CURRICULAR:\n{topico_selecionado['texto_completo']}"
                mensagens_para_api = [{"role": "system", "content": system_prompt}] + st.session_state.messages
                try:
                    response = client.chat.completions.create(model="gpt-4o-mini", messages=mensagens_para_api)
                    resposta_texto = response.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": resposta_texto})
                except Exception as e:
                    st.error(f"Ocorreu um erro com a API da OpenAI: {e}")
        
        perguntas_do_usuario = [msg for msg in st.session_state.messages if msg['role'] == 'user']
        gatilho = 3 
        if len(perguntas_do_usuario) > 0 and len(perguntas_do_usuario) % gatilho == 0:
            if st.session_state.analise_feita_para_pergunta != len(perguntas_do_usuario):
                sugestao = analisar_progresso_do_topico(
                    historico=st.session_state.messages,
                    topico_atual_idx=st.session_state.topico_selecionado_idx,
                    df_curriculo=df,
                    client=client,
                    topicos_superados=st.session_state.topicos_superados
                )
                if sugestao:
                    acao = sugestao.get('acao_sugerida')
                    if acao == 'avancar' and sugestao.get('proximo_topico_sugerido'):
                        st.session_state.sugestao_pendente = sugestao
                        st.session_state.desempenho_status = "avancar"
                    elif acao in ['continuar', 'revisar']:
                        st.session_state.desempenho_status = "continuar"
                    st.session_state.analise_feita_para_pergunta = len(perguntas_do_usuario)
        st.rerun()

# --- 5. SIDEBAR (SEMPRE VISÍVEL) ---
with st.sidebar:
    st.header("Debug")
    with st.expander("🔌 Terminal de Debug", expanded=True):
        log_container = st.container(height=300)
        log_text = "\n".join(st.session_state.log_messages)
        log_container.text(log_text)
    
    st.divider()
    
    status = st.session_state.get('desempenho_status')
    if status == 'analisando':
        st.warning("Desempenho: Analisando...")
    elif status == 'continuar':
        st.error("Desempenho: Continua na lição")
    elif status == 'avancar':
        st.success("Desempenho: Pronto para o próximo tópico")
