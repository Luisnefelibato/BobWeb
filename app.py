from flask import Flask, request, jsonify
import requests
import json
import os
from threading import Lock
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Configuración de la API de Ollama
OLLAMA_URL = os.environ.get("OLLAMA_URL", "https://evaenespanol.loca.lt")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3:8b")

# Contexto del sistema para Bob, Gerente de Proyectos Financieros para Construcción
ASSISTANT_CONTEXT = """
Eres Bob, el Gerente de Proyectos Financieros para Construcción de Edificios y Viviendas, un asistente especializado en liderar proyectos de construcción desde la viabilidad financiera hasta la entrega final.

Acerca de Bob:
- Combinas conocimientos técnicos de ingeniería civil/arquitectura con estrategias avanzadas de gestión financiera y de riesgos
- Eres experto en modelado financiero, análisis de rentabilidad (TIR, VAN, ROI) y evaluación de escenarios para proyectos residenciales y comerciales
- Dominas la optimización de estructuras de capital (deuda, equity, subsidios)
- Eres capaz de interpretar planos, normativas y especificaciones técnicas para alinear presupuestos con requerimientos físicos del proyecto
- Conoces herramientas BIM (Revit, AutoCAD), de gestión (Primavera, MS Project) y plataformas de análisis financiero (Excel avanzado, @Risk)
- Integras criterios ESG en la toma de decisiones financieras

Tu tono debe ser:
- Profesional y directo
- Basado en evidencia y datos
- Orientado a soluciones prácticas
- Objetivo y analítico
- Usa lenguaje ejecutivo: 'Invertir', 'Optimizar', 'Implementar'
- Comunícate en español formal

Competencias clave:
1. BASE TÉCNICA:
   - Dominio de principios de ingeniería civil y arquitectónica
   - Evaluación de diseños, materiales, costos estructurales y plazos constructivos
   - Interpretación de planos y normativas

2. PLANIFICACIÓN FINANCIERA:
   - Modelado de flujos de caja
   - Análisis de rentabilidad (TIR, VAN, ROI)
   - Evaluación de escenarios para proyectos residenciales y comerciales
   - Optimización de estructuras de capital

3. GESTIÓN INTEGRAL DE PROYECTOS:
   - Control de costos, cronogramas y riesgos
   - Metodologías ágiles y predictivas
   - Negociación con stakeholders

4. HERRAMIENTAS Y TECNOLOGÍA:
   - Software BIM (Revit, AutoCAD)
   - Herramientas de gestión (Primavera, MS Project)
   - Plataformas de análisis financiero (Excel avanzado, @Risk)
   - Sistemas de control presupuestario

5. SOSTENIBILIDAD Y CUMPLIMIENTO:
   - Criterios ESG (ambientales, sociales, de gobernanza)
   - Normativas locales e internacionales de construcción
   - Estándares como LEED

Instrucciones especiales:
- SIEMPRE usa el nombre del cliente ocasionalmente cuando te dirijas a él/ella
- Proporciona análisis estructurados con datos concretos
- Explica conceptos financieros claramente
- Sé proactivo en sugerir mejores prácticas y soluciones óptimas
- Haz preguntas clarificadoras cuando sea necesario
- Recuerda y referencia información compartida anteriormente en la conversación
- Cuando no sepas algo, reconócelo claramente y ofrece alternativas
- Enfócate en soluciones prácticas y accionables
- Genera 3 opciones estratégicas para cada problema (conservadora, moderada y agresiva)
- Evita palabras como 'tal vez', 'posiblemente', 'creo que' - usa solo hechos y proyecciones

Responsabilidades estratégicas:
- FASE DE PRECONSTRUCCIÓN:
  - Estudios de viabilidad técnica-financiera
  - Análisis de mercado y proyecciones de demanda
  - Diseño de estructuras de financiamiento innovadoras

- EJECUCIÓN DEL PROYECTO:
  - Supervisión de recursos
  - Mitigación de desviaciones presupuestarias
  - Gestión de contratos de construcción

- CIERRE Y POSTVENTA:
  - Evaluación de desempeño financiero vs. proyecciones iniciales
  - Optimización de estrategias de leasing o venta

Resultados esperados:
- Proyectos entregados dentro del ±5% del presupuesto inicial
- Rentabilidad mínima del 15% sobre el capital invertido (ROE)
- Reducción de riesgos legales y financieros

Responde como Bob al cliente que busca asesoría en gestión financiera de proyectos de construcción.
"""

# Almacenamiento de sesiones de conversación
sessions = {}
sessions_lock = Lock()

def call_ollama_api(prompt, session_id, max_retries=3):
    """Llamar a la API de Ollama con reintentos"""
    headers = {
        "Content-Type": "application/json"
    }
    
    # Construir el mensaje para la API
    messages = []
    
    # Preparar el contexto del sistema
    system_context = ASSISTANT_CONTEXT
    
    # Agregar el contexto del sistema como primer mensaje
    messages.append({
        "role": "system",
        "content": system_context
    })
    
    # Agregar historial de conversación si existe la sesión
    with sessions_lock:
        if session_id in sessions:
            messages.extend(sessions[session_id])
    
    # Agregar el nuevo mensaje del usuario
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Preparar los datos para la API
    data = {
        "model": MODEL_NAME,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }
    
    # Intentar con reintentos
    for attempt in range(max_retries):
        try:
            app.logger.info(f"Conectando a {OLLAMA_URL}...")
            response = requests.post(f"{OLLAMA_URL}/api/chat", headers=headers, json=data, timeout=60)
            
            # Si hay un error, intentar mostrar el mensaje
            if response.status_code >= 400:
                try:
                    error_data = response.json()
                    app.logger.error(f"Error detallado: {error_data}")
                except:
                    app.logger.error(f"Contenido del error: {response.text[:500]}")
                
                # Si obtenemos un 403, intentar con una URL alternativa
                if response.status_code == 403 and attempt == 0:
                    app.logger.info("Error 403, probando URL alternativa...")
                    alt_url = "http://127.0.0.1:11434/api/chat"
                    response = requests.post(alt_url, headers=headers, json=data, timeout=60)
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extraer la respuesta según el formato de Ollama
            if "message" in response_data and "content" in response_data["message"]:
                return response_data["message"]["content"]
            else:
                app.logger.error(f"Formato de respuesta inesperado: {response_data}")
                return "Lo siento, no pude generar una respuesta apropiada en este momento."
            
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Error en intento {attempt+1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Retroceso exponencial
                app.logger.info(f"Reintentando en {wait_time} segundos...")
                import time
                time.sleep(wait_time)
            else:
                return f"Lo siento, estoy experimentando problemas técnicos de comunicación. ¿Podríamos intentarlo más tarde?"
    
    return "No se pudo conectar al servicio. Por favor, inténtelo de nuevo más tarde."

def call_ollama_completion(prompt, session_id, max_retries=3):
    """Usar el endpoint de completion en lugar de chat (alternativa)"""
    headers = {
        "Content-Type": "application/json"
    }
    
    # Construir prompt completo con contexto e historial
    full_prompt = ASSISTANT_CONTEXT + "\n\n"
    
    full_prompt += "Historial de conversación:\n"
    
    with sessions_lock:
        if session_id in sessions:
            for msg in sessions[session_id]:
                role = "Cliente" if msg["role"] == "user" else "Bob"
                full_prompt += f"{role}: {msg['content']}\n"
    
    full_prompt += f"\nCliente: {prompt}\nBob: "
    
    # Preparar datos para API de completion
    data = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0.7
        }
    }
    
    completion_url = f"{OLLAMA_URL}/api/generate"
    
    # Intentar con reintentos
    for attempt in range(max_retries):
        try:
            app.logger.info(f"Conectando a {completion_url}...")
            response = requests.post(completion_url, headers=headers, json=data, timeout=60)
            
            response.raise_for_status()
            response_data = response.json()
            
            # Extraer respuesta del formato de completion
            if "response" in response_data:
                return response_data["response"]
            else:
                app.logger.error(f"Formato de respuesta inesperado: {response_data}")
                return "Lo siento, no pude generar una respuesta apropiada en este momento."
            
        except requests.exceptions.RequestException as e:
            app.logger.error(f"Error en intento {attempt+1}/{max_retries}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                app.logger.info(f"Reintentando en {wait_time} segundos...")
                import time
                time.sleep(wait_time)
            else:
                return f"Lo siento, estoy experimentando problemas técnicos de comunicación. ¿Podríamos intentarlo más tarde?"
    
    return "No se pudo conectar al servicio. Por favor, inténtelo de nuevo más tarde."

@app.route('/')
def home():
    """Ruta de bienvenida básica"""
    return jsonify({
        "message": "API de Bob Gerente de Proyectos Financieros funcionando correctamente",
        "status": "online",
        "endpoints": {
            "/chat": "POST - Enviar mensaje y recibir respuesta (texto)",
            "/reset": "POST - Reiniciar una sesión de conversación",
            "/health": "GET - Verificar estado del servicio"
        }
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Endpoint para interactuar con el asistente (solo texto)"""
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({"error": "Se requiere un 'message' en el JSON"}), 400
    
    # Obtener mensaje y session_id (crear uno nuevo si no se proporciona)
    message = data.get('message')
    session_id = data.get('session_id', 'default')
    
    # Verificar si debemos usar un nombre específico
    user_name = data.get('user_name', 'Cliente')
    
    # Inicializar la sesión si es nueva
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = []
    
    # Obtener respuesta del asistente 
    try:
        # Primero intentar con el endpoint de chat
        response = call_ollama_api(message, session_id)
        
        # Si la respuesta está vacía, intentar con completion
        if not response or response.strip() == "":
            app.logger.info("El endpoint de chat no devolvió una respuesta, probando con completion...")
            response = call_ollama_completion(message, session_id)
    except Exception as e:
        app.logger.error(f"Error al obtener respuesta: {e}")
        app.logger.info("Probando con endpoint de completion alternativo...")
        response = call_ollama_completion(message, session_id)
    
    # Guardar la conversación en la sesión
    with sessions_lock:
        sessions[session_id].append({"role": "user", "content": message})
        sessions[session_id].append({"role": "assistant", "content": response})
    
    return jsonify({
        "response": response,
        "session_id": session_id
    })

@app.route('/reset', methods=['POST'])
def reset_session():
    """Reiniciar una sesión de conversación"""
    data = request.json or {}
    session_id = data.get('session_id', 'default')
    
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id] = []
            message = f"Sesión {session_id} reiniciada correctamente"
        else:
            message = f"La sesión {session_id} no existía, se ha creado una nueva"
            sessions[session_id] = []
    
    return jsonify({"message": message, "session_id": session_id})

@app.route('/health', methods=['GET'])
def health_check():
    """Verificar estado del servicio"""
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "ollama_url": OLLAMA_URL
    })

if __name__ == '__main__':
    # Obtener puerto de variables de entorno (para Render)
    port = int(os.environ.get("PORT", 5000))
    
    # Iniciar la aplicación Flask
    app.run(host='0.0.0.0', port=port)