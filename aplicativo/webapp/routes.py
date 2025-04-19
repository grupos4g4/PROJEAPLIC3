import logging
import time
from flask import Blueprint, render_template, request, jsonify
from PIL import Image
from webapp.model.predict import classify_image

# Criando o Blueprint para as rotas principais
main = Blueprint('main', __name__)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@main.route('/')
def home():
    """Renderiza a página inicial."""
    return render_template('index.html')

@main.route('/classify', methods=['POST'])
def classify():
    """Processa a imagem enviada e retorna o resultado da classificação."""
    try:
        logger.info("Recebendo requisição para classificação de imagem")
        start_time = time.time()

        # Verifica se um arquivo foi enviado na requisição
        if 'image' not in request.files:
            logger.warning("Nenhuma imagem foi enviada na requisição")
            return jsonify({'error': 'Nenhuma imagem enviada'}), 400

        file = request.files['image']

        # Verifica se o nome do arquivo está vazio
        if file.filename == '':
            logger.warning("Nenhum arquivo selecionado")
            return jsonify({'error': 'Nenhum arquivo selecionado'}), 400

        # Abre a imagem e redimensiona se necessário
        try:
            image = Image.open(file.stream)
            max_size = (800, 800)
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
        except Exception as e:
            logger.error(f"Erro ao abrir a imagem: {str(e)}")
            return jsonify({'error': 'Erro ao processar a imagem. Envie um arquivo válido.'}), 400

        logger.info("Classificando a imagem...")
        results = classify_image(image)
        logger.info(f"Classificação concluída em {time.time() - start_time:.2f}s")

        # Renderiza a página de resultado com as previsões
        return render_template('result.html', results=results)

    except Exception as e:
        logger.error(f"Erro inesperado durante a classificação: {str(e)}")
        return jsonify({'error': 'Erro interno no servidor. Tente novamente.'}), 500
