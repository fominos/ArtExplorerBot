import torch
import dill
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# Устройство для загрузки моделей (CPU или GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class EmotionModel(nn.Module):
    def __init__(
        self,
        num_emotion_classes=8,
        num_style_classes=27,
        device="cpu",
        num_layers=1,
        use_dropout=False,
        dropout=0.5
    ):
        super(EmotionModel, self).__init__()
        self.device = device
        self.num_emotion_classes = num_emotion_classes
        self.num_style_classes = num_style_classes
        self.num_layers = num_layers
        self.use_dropout = use_dropout
        self.dropout = dropout

        # Инициализация MobileNetV3 Large
        self.cnn = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = 960  # Размерность выходных признаков MobileNetV3 Large после удаления классификатора
        self.cnn.classifier = nn.Identity()  # Удаляем последний слой

        # Инициализация классификаторов
        self.emotion_classifier = self._build_classifier(in_features, self.num_emotion_classes)
        self.style_classifier = self._build_classifier(in_features, self.num_style_classes)

        # Инициализация весов
        self._initialize_weights()

    def _build_classifier(self, in_features, num_classes):
        """Создание классификатора с заданным количеством слоев и Dropout."""
        layers = []
        for _ in range(self.num_layers):
            layers.append(nn.Linear(in_features, 512))
            layers.append(nn.ReLU())
            if self.use_dropout:
                layers.append(nn.Dropout(p=self.dropout))
            in_features = 512  # Обновляем количество входных признаков

        # Выходной слой
        layers.append(nn.Linear(in_features, num_classes))

        return nn.Sequential(*layers).to(self.device)

    def _initialize_weights(self):
        """Инициализация весов новых слоев."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images):
        """Прямой проход через модель."""
        # Извлечение признаков изображения
        features = self.cnn(images)

        # Классификация эмоций
        emotion_output = self.emotion_classifier(features)

        # Классификация стилей
        style_output = self.style_classifier(features)

        return emotion_output, style_output

# Загрузка модели для классификации
def load_classification_model(model_path):
    try:
        with open(model_path, "rb") as f:
            model = dill.load(f)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели классификации: {e}")
        return None

# Загрузка модели для анализа эмоций
def load_emotion_model(model_path, num_emotion_classes=8, num_style_classes=27):
    try:
        # Создайте экземпляр вашей модели с нужным количеством классов
        model = EmotionModel(num_emotion_classes=num_emotion_classes, num_style_classes=num_style_classes)

        # Загрузите состояние модели
        state_dict = torch.load(model_path)

        # Фильтруем параметры, которые не соответствуют
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.size() == model.state_dict()[k].size()}

        # Загружаем отфильтрованные параметры
        model.load_state_dict(filtered_state_dict, strict=False)
        model.to(model.device)
        model.eval()
        return model
    except Exception as e:
        print(f"Ошибка при загрузке модели эмоций: {e}")
        return None

# Загрузка модели BLIP для генерации описаний
def load_blip_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            model, processor = dill.load(f)
        model.to(device)
        return model, processor
    except Exception as e:
        print(f"Ошибка при загрузке модели BLIP: {e}")
        return None, None

# Преобразование изображения для модели
def preprocess_image(image_path, input_size=224):
    try:
        transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return None