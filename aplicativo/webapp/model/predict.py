import torch
import torchvision.transforms as transforms
from torchvision import models
from pathlib import Path
import gdown  # Para baixar arquivos do Google Drive
import pandas as pd 

# Diretórios do projeto
csv_path = Path("webapp/model/tabela-nutricional-frutas.csv") 
model_path = Path("webapp/model/modelo_pesos.pth")  

# Link do Google Drive (novo modelo atualizado)
gdrive_url = "https://drive.google.com/uc?id=1--J64FPpV0ig3OfqTLZultWQj7xUoi7w"

# Baixar pesos do Google Drive, se não existir
if not model_path.exists():
    print("Pesos do modelo não encontrados. Baixando do Google Drive...")
    try:
        gdown.download(gdrive_url, str(model_path), fuzzy=True, quiet=False)
        print(f"Modelo salvo em: {model_path}")
    except Exception as e:
        raise FileNotFoundError(f"Erro ao baixar o modelo: {e}")

if not model_path.exists():
    raise FileNotFoundError(f"Erro: O arquivo de pesos do modelo não foi encontrado: {model_path}")

# Carregar tabela nutricional
if csv_path.exists():
    df_nutricional = pd.read_csv(csv_path, delimiter=";")
    class_names = df_nutricional["Fruta (100g)"].tolist()
else:
    raise FileNotFoundError(f"Erro: O arquivo CSV não foi encontrado: {csv_path}")

num_classes = len(class_names)

# Cache do modelo
_model = None

def load_model():
    """Carrega o modelo ResNet50 com os pesos treinados."""
    print(f"Carregando modelo de: {model_path}")  # Log para depuração
    
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)

    try:
        model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))
        model.eval()
        print("Modelo carregado com sucesso!")
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo: {e}")

    return model

def get_model():
    """Retorna o modelo carregado, usando cache para eficiência."""
    global _model
    if _model is None:
        _model = load_model()
    return _model

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_nutritional_info(fruit_name):
    """Busca informações nutricionais da fruta na tabela."""
    info = df_nutricional[df_nutricional["Fruta (100g)"] == fruit_name]
    if not info.empty:
        return info.iloc[0].to_dict()  # Retorna os dados nutricionais como um dicionário
    return None

def classify_image(image):
    """Recebe uma imagem PIL, faz a predição e retorna as 3 classes prováveis com tabela nutricional."""
    model = get_model()  # Usa modelo em cache
    image = preprocess(image).unsqueeze(0) 

    with torch.no_grad():  # Desativa cálculo de gradientes
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]  # Probabilidades por classe
    
    # Pegando as 3 classes com maior probabilidade
    top_probs, top_catids = torch.topk(probabilities, 3)

    # Lista de resultados
    results = []
    for i in range(3):
        fruit_name = class_names[top_catids[i].item()]
        nutritional_info = get_nutritional_info(fruit_name)

        results.append({
            "fruit_name": fruit_name,
            "probability": f"{top_probs[i].item() * 100:.2f}%",
            "nutritional_info": nutritional_info
        })

    return results
