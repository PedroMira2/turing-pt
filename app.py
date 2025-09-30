from flask import Flask, render_template, request, jsonify, session
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import random
import time
import uuid
from datetime import datetime
import os

app = Flask(__name__)
app.secret_key = 'turing_test_secret_key_2024'

# Inicializa o modelo (vai baixar na primeira execução)
print("🚀 Carregando modelo de IA...")
try:
    chatbot = pipeline(
        "text-generation",
        model="microsoft/DialoGPT-small",
        tokenizer="microsoft/DialoGPT-small",
        torch_dtype=torch.float32
    )
    MODEL_LOADED = True
    print("✅ Modelo carregado com sucesso!")
except Exception as e:
    print(f"❌ Erro ao carregar modelo: {e}")
    MODEL_LOADED = False
    chatbot = None

class TuringSession:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.is_human = random.choice([True, False])
        self.messages = []
        self.start_time = datetime.now()
        self.message_count = 0
        self.add_message("system", "Olá! Vamos conversar? Pode me fazer qualquer pergunta!")
    
    def add_message(self, sender, text):
        self.messages.append({
            'sender': sender,
            'text': text,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        if sender == 'user':
            self.message_count += 1

def get_ai_response(user_message):
    if not MODEL_LOADED or chatbot is None:
        return "Estou com problemas técnicos. Podemos continuar mais tarde?"
    
    try:
        # Limita o tamanho da entrada para evitar problemas
        if len(user_message) > 500:
            user_message = user_message[:500]
        
        response = chatbot(
            user_message,
            max_length=150,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=50256,
            repetition_penalty=1.1
        )
        
        # Limpa a resposta
        ai_response = response[0]['generated_text'].strip()
        
        # Remove a mensagem do usuário da resposta se estiver incluída
        if user_message in ai_response:
            ai_response = ai_response.replace(user_message, "").strip()
        
        # Garante que a resposta não esteja vazia
        if not ai_response or len(ai_response) < 2:
            ai_response = "Interessante! Pode me contar mais sobre isso?"
            
        return ai_response
        
    except Exception as e:
        print(f"Erro na geração: {e}")
        return "Hmm, não sei bem o que dizer sobre isso. O que mais você gostaria de conversar?"

def get_human_response(user_message):
    human_responses = [
        "Interessante! Conte-me mais sobre isso...",
        "Nunca tinha pensado por esse ângulo! O que te fez pensar nisso?",
        "Isso me lembra algo que aconteceu comigo recentemente!",
        "Hmm, não tenho certeza sobre isso. Qual sua opinião?",
        "Que legal! Também me interesso por esse assunto!",
        "Às vezes me pego pensando nisso também... é complicado!",
        "Você já teve alguma experiência relacionada a isso?",
        "Puxa, isso é algo que dá pano pra manga!",
        "Não sei muito sobre isso, mas achei fascinante!",
        "Como você se sente em relação a isso?",
        "Isso é realmente algo para refletir...",
        "Já passei por situação parecida! Foi bem desafiador!",
        "Não tenho uma opinião formada, mas acho intrigante!",
        "Que conversa interessante estamos tendo!",
        "Isso me faz pensar sobre muitas coisas...",
        "As vezes a vida nos surpreende, não é?",
        "Você acha que isso é comum para a maioria das pessoas?",
        "Me conta mais, estou curioso!",
        "Nossa, nunca tinha visto por essa perspectiva!",
        "Isso é algo que sempre me questionei também!"
    ]
    return random.choice(human_responses)

@app.route('/')
def index():
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    session['turing_session'] = TuringSession()
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    if 'turing_session' not in session:
        return jsonify({'error': 'Sessão não encontrada'})
    
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Mensagem vazia'})
    
    turing_session = session['turing_session']
    turing_session.add_message('user', user_message)
    
    # Simula tempo de resposta humana/IA (1-3 segundos)
    response_delay = random.uniform(1.0, 3.0)
    time.sleep(response_delay)
    
    if turing_session.is_human:
        response = get_human_response(user_message)
    else:
        response = get_ai_response(user_message)
    
    turing_session.add_message('agent', response)
    
    return jsonify({
        'response': response,
        'message_count': turing_session.message_count,
        'is_typing': False
    })

@app.route('/reveal', methods=['POST'])
def reveal():
    if 'turing_session' not in session:
        return jsonify({'error': 'Sessão não encontrada'})
    
    turing_session = session['turing_session']
    
    # Calcula tempo total
    end_time = datetime.now()
    duration = end_time - turing_session.start_time
    minutes = int(duration.total_seconds() // 60)
    seconds = int(duration.total_seconds() % 60)
    
    identity = "humano" if turing_session.is_human else "IA"
    
    # Estatísticas
    stats = {
        'identity': identity,
        'message_count': turing_session.message_count,
        'duration': f"{minutes:02d}:{seconds:02d}",
        'conversation': turing_session.messages
    }
    
    return jsonify(stats)

@app.route('/new_session', methods=['POST'])
def new_session():
    session['turing_session'] = TuringSession()
    return jsonify({'status': 'Nova sessão iniciada'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)