# ESTADO 3: CHAT (LÓGICA UNIFICADA E CORRIGIDA)
elif st.session_state.app_state == "CHAT":
    # Botão para voltar para a seleção de tópicos
    if st.button("⬅️ Mudar de Tópico"):
        st.session_state.messages = []
        st.session_state.app_state = "SELECAO_TOPICO"
        st.session_state.df_com_similaridade = None # Limpa para forçar novo cálculo
        st.rerun()

    # Exibe o histórico da conversa
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            renderizar_mensagem(message)
    
    # Lógica de input (botões ou caixa de texto)
    prompt_gerado = None
    if not st.session_state.get("initial_action_taken", False):
        st.write("Escolha uma ação:")
        col1, col2, col3 = st.columns(3)
        
        if col1.button("Explique o tópico", use_container_width=True):
            prompt_gerado = "Por favor, me dê uma explicação detalhada sobre este tópico."
        if col2.button("Me dê um exemplo", use_container_width=True):
            prompt_gerado = "Pode me dar um exemplo prático sobre isso?"
        if col3.button("Quero exercícios", use_container_width=True):
            prompt_gerado = "Gostaria de alguns exercícios para praticar."

        if prompt_gerado:
            st.session_state.initial_action_taken = True
    else:
        # Se a ação inicial já foi tomada, mostra a caixa de texto
        prompt_gerado = st.chat_input("Faça outra pergunta ou peça mais exercícios!")

    # Se um prompt foi gerado (seja por botão ou por texto), processe-o
    if prompt_gerado:
        st.session_state.messages.append({"role": "user", "content": prompt_gerado})
        # Roda o processamento da API imediatamente após a entrada do usuário
        with st.chat_message("user"):
             st.markdown(prompt_gerado)
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                st.session_state.log_messages = []
                log_to_terminal("--- NOVA QUERY (CHAT) ---")
                
                topico_atual_texto = df.loc[st.session_state.topico_selecionado_idx, 'texto_completo']
                query_para_rag = criar_query_contextualizada(st.session_state.messages, topico_atual_texto)
                
                # --- ESTE É O SYSTEM PROMPT COMPLETO E CORRETO ---
                system_prompt = f"""
                Você é um tutor de matemática. Sua resposta DEVE ser um objeto JSON válido com uma chave "response" contendo uma lista de strings.
                Use Markdown e LaTeX (com $...$) para formatar o texto dentro das strings.

                Exemplo de resposta JSON válida:
                {{
                  "response": [
                    "A fórmula de Bhaskara é usada para resolver equações de segundo grau.",
                    "A fórmula é: $$\\Delta = b^2 - 4ac$$",
                    "- Onde $a$, $b$, e $c$ são os coeficientes da equação."
                  ]
                }}

                O aluno está estudando o tópico: "{df.loc[st.session_state.topico_selecionado_idx, 'Objetos do conhecimento']}".
                Use o CONTEXTO CURRICULAR abaixo para responder a pergunta dele, seguindo ESTRITAMENTE o formato JSON.
                CONTEXTO CURRICULAR: {topico_atual_texto}
                """
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
        st.rerun()
