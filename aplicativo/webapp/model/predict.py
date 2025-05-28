import torch
import torchvision.transforms as transforms
from torchvision import models
from pathlib import Path
import gdown  # Para baixar arquivos do Google Drive
import pandas as pd
from scipy.spatial.distance import euclidean

# Diretórios do projeto
csv_path = Path("webapp/model/tabela-nutricional-frutas.csv") 
model_path = Path("webapp/model/modelo_pesos.pth")  
gdrive_url = "https://drive.google.com/uc?id=1r713aBkYORtJsTb3XihRg0AUoWWL005q"

def download_model():
    """Baixa os pesos do modelo do Google Drive se não existirem."""
    if not model_path.exists():
        print("Pesos do modelo não encontrados. Baixando do Google Drive...")
        try:
            gdown.download(gdrive_url, str(model_path), fuzzy=True, quiet=False)
            print(f"Modelo salvo em: {model_path}")
        except Exception as e:
            raise FileNotFoundError(f"Erro ao baixar o modelo: {e}")

def load_nutrition_data():
    """Carrega o arquivo CSV contendo os dados nutricionais."""
    if csv_path.exists():
        df = pd.read_csv(csv_path, delimiter=";")
        return df
    else:
        raise FileNotFoundError(f"Erro: O arquivo CSV não foi encontrado: {csv_path}")

download_model()
df_nutricional = load_nutrition_data()
class_names = df_nutricional["Fruta (100g)"].tolist()
num_classes = len(class_names)

# Função para carregar o modelo
_model = None

def load_model():
    """Carrega o modelo ResNet50 com os pesos treinados."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    
    try:
        model.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Erro ao carregar o modelo: {e}")

    return model

def get_model():
    """Retorna o modelo carregado"""
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
    """Busca informações nutricionais na tabela."""
    info = df_nutricional[df_nutricional["Fruta (100g)"] == fruit_name]
    if not info.empty:
        return info.iloc[0].to_dict()
    return None

def classify_image(image):
    """Recebe uma imagem faz a predição e retorna as 3 classes prováveis com tabela nutricional e recomendação apenas para a mais provável."""
    model = get_model()
    image = preprocess(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    top_probs, top_catids = torch.topk(probabilities, 3)
    results = []
    
    for i in range(3):
        fruit_name = class_names[top_catids[i].item()]
        nutritional_info = get_nutritional_info(fruit_name)
        recommended_fruit = recommend_similar_fruit(fruit_name) if i == 0 else None  # Apenas para a fruta mais provável
        recommended_info = get_nutritional_info(recommended_fruit) if recommended_fruit else None

        results.append({
            "fruit_name": fruit_name,
            "probability": f"{top_probs[i].item() * 100:.2f}%",
            "nutritional_info": nutritional_info,
            "recommended_fruit": recommended_fruit,
            "recommended_info": recommended_info
        })
    
    return results

def recommend_similar_fruit(fruit_name):
    """Recomenda uma fruta com informações nutricionais mais próximas da fruta identificada."""
    if fruit_name not in df_nutricional["Fruta (100g)"].values:
        return f"Fruta {fruit_name} não encontrada no banco de dados."
    
    fruit_info = df_nutricional[df_nutricional["Fruta (100g)"] == fruit_name].iloc[0, 1:].values
    
    min_distance = float('inf')
    closest_fruit = None
    
    for _, row in df_nutricional.iterrows():
        candidate_fruit = row["Fruta (100g)"]
        if candidate_fruit == fruit_name:
            continue
        
        candidate_info = row.iloc[1:].values
        distance = euclidean(fruit_info, candidate_info)
        
        if distance < min_distance:
            min_distance = distance
            closest_fruit = candidate_fruit
    
    return closest_fruit