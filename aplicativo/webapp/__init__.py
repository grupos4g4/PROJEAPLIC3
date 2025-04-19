import logging
from flask import Flask
from webapp.routes import main

# Configuração de logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_app():
    """Cria e configura a aplicação Flask."""
    logger.info("Iniciando aplicação...")
    app = Flask(__name__)
    # Registra as rotas
    try:
        app.register_blueprint(main)
        logger.info("Blueprint registrado com sucesso")
    except Exception as e:
        logger.error(f"Erro ao registrar blueprint: {e}")
        raise

    return app
