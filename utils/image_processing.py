import torch
from PIL import Image
from app.utils.model_utils import load_classification_model, load_emotion_model, load_blip_model, preprocess_image
import traceback

# Классификация изображения (реальное или AI-генерированное)
def classify_image(image_path, model_path):
    model = load_classification_model(model_path)
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = torch.sigmoid(output).item()
        print(prediction)
    return "Создано ИИ" if prediction < 0.5 else "Реальное"

# Анализ стиля и эмоций
def analyze_style_and_emotion(image_path, model_path):
    torch.autograd.set_detect_anomaly(True)
    # Загрузка модели
    model = load_emotion_model(model_path)
    if model is None:
        raise ValueError("Модель не загружена!")

    # Преобразование изображения
    image_tensor = preprocess_image(image_path)
    if image_tensor is None:
        raise ValueError("Ошибка при обработке изображения!")


    # Перемещение данных на то же устройство, что и модель
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    # Проверка на NaN или inf во входных данных
    if torch.isnan(image_tensor).any() or torch.isinf(image_tensor).any():
        raise ValueError("Входные данные содержат NaN или inf!")

    # Классы для эмоций и стилей
    """emotion_classes = ['sadness', 'contentment', 'awe', 'amusement', 'excitement', 'fear', 'disgust', 'anger']
    style_classes = ['Abstract_Expressionism', 'Action_painting', 'Analytical_Cubism', 'Art_Nouveau_Modern', 'Baroque',
                    'Color_Field_Painting', 'Contemporary_Realism', 'Cubism', 'Early_Renaissance', 'Expressionism', 'Fauvism', 'High_Renaissance',
                    'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism', 'Naive_Art_Primitivism', 'New_Realism',
                    'Northern_Renaissance', 'Pointillism', 'Pop_Art', 'Post_Impressionism', 'Realism', 'Rococo',
                    'Romanticism', 'Symbolism', 'Synthetic_Cubism', 'Ukiyo_e']"""
    emotion_classes = ['Грусть', 'Умиротворение', 'Восхищение', 'Веселье', 'Восторг', 'Страх', 'Отвращение', 'Гнев']
    style_classes = ['Абстрактный экспрессионизм', 'Живопись действия', 'Аналитический кубизм', 'Модерн', 'Барокко',
                     'Живопись цветового поля', 'Современный реализм', 'Кубизм', 'Раннее возрождение', 'Экспрессионизм', 'Фовизм',
                     'Высокое Возрождение', 'Импрессионизм', 'Маньеризм / Позднее Возрождение', 'Минимализм',
                     'Наивное искусство / Примитивизм', 'Новый реализм', 'Северное Возрождение', 'Пуантилизм', 'Поп-арт',
                     'Постимпрессионизм', 'Реализм', 'Рококо','Романтизм', 'Символизм', 'Синтетический кубизм', 'Укиё-э (японская гравюра)']
    # Передача изображения в модель
    with torch.no_grad():  # Отключаем градиенты для предсказания
        try:
            emotion_output, style_output = model(image_tensor)
        except Exception as e:
            print(f"Ошибка при выполнении модели: {e}")
            traceback.print_exc()
            raise

    # Проверка на NaN или inf в выходных данных
    if torch.isnan(emotion_output).any() or torch.isinf(emotion_output).any():
        raise ValueError("Emotion output содержит NaN или inf!")
    if torch.isnan(style_output).any() or torch.isinf(style_output).any():
        raise ValueError("Style output содержит NaN или inf!")

    # Преобразование логитов в вероятности
    emotion_probs = torch.softmax(emotion_output, dim=1)
    style_probs = torch.softmax(style_output, dim=1)
    print(style_probs)
    print(emotion_probs)
    # Получение индексов классов с максимальной вероятностью для стиля
    predicted_style_idx = torch.argmax(style_probs, dim=1).item()

    # Получение названий классов для стиля
    predicted_style = style_classes[predicted_style_idx]

    # Устанавливаем порог вероятности (например, 0.1)
    threshold = 0.2

    # Определяем, какие эмоции превышают порог
    above_threshold_mask = emotion_probs > threshold

    # Получаем индексы строк и столбцов для эмоций, превышающих порог
    row_indices, col_indices = above_threshold_mask.nonzero(as_tuple=True)

    # Получаем вероятности предсказанных эмоций
    predicted_emotions_probs = emotion_probs[row_indices, col_indices]
    print(predicted_emotions_probs)
    # Если есть эмоции, превышающие порог
    if len(predicted_emotions_probs) > 0:
        # Сортируем индексы предсказанных эмоций по убыванию вероятностей
        sorted_indices = predicted_emotions_probs.argsort(descending=True)

        # Создаем строку с предсказанными эмоциями
        predicted_emotions = ""
        for i, idx in enumerate(sorted_indices):
            emotion_idx = col_indices[idx]  # Индекс эмоции, соответствующей вероятности

            if i == 0:
                # Самая сильная эмоция
                predicted_emotions += f"{emotion_classes[emotion_idx]}"
            elif i == 1:
                # Остальные эмоции
                predicted_emotions += f", так же может вызывать {emotion_classes[emotion_idx]}"
            else:
                predicted_emotions += f", {emotion_classes[emotion_idx]}"
    else:
        # Если ни одна эмоция не превышает порог, выводим самую сильную эмоцию
        strongest_emotion_idx = emotion_probs.argmax(dim=1).item()  # Индекс самой сильной эмоции
        predicted_emotions = f"{emotion_classes[strongest_emotion_idx]}"

    return predicted_style, predicted_emotions

# Генерация описания
def generate_description(image_path, model_path, model_path_original):
    model, processor = load_blip_model(model_path)
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs)
    description = processor.decode(outputs[0], skip_special_tokens=True)
    model_or, processor_or = load_blip_model(model_path_original)
    inputs = processor_or(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model_or.generate(**inputs)
    description_or = processor_or.decode(outputs[0], skip_special_tokens=True)
    description_final = description_or + ". " + description
    return description_final