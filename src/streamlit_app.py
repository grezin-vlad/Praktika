import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

import json
import os
from datetime import datetime
import pandas as pd


# =========================
# НАСТРОЙКИ
# =========================

st.set_page_config(
    page_title="EuroSAT AI",
    layout="wide"
)


# =========================
# КЛАССЫ
# =========================

CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake"
]


# =========================
# ЗАГРУЗКА МОДЕЛИ
# =========================

@st.cache_resource
def load_model():

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        model.classifier[1].in_features,
        10
    )


    model.load_state_dict(
        torch.load(
            "models/efficientnet_b0.pth",
            map_location="cpu"
        )
    )


    model.eval()

    return model



model = load_model()



# =========================
# ОБРАБОТКА ИЗОБРАЖЕНИЯ
# =========================

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])



# =========================
# HISTORY JSON
# =========================

def save_result(
        filename,
        predicted_class,
        confidence
):

    os.makedirs(
        "results",
        exist_ok=True
    )


    new_result = {

        "time":
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "file":
        filename,

        "class":
        predicted_class,

        "confidence":
        round(confidence,2)

    }



    history_file = (
        "results/history.json"
    )


    if os.path.exists(history_file):

        try:

            with open(
                history_file,
                "r",
                encoding="utf-8"
            ) as f:

                history = json.load(f)


        except:

            history = []


    else:

        history = []



    history.append(
        new_result
    )



    with open(
        history_file,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            history,
            f,
            indent=4,
            ensure_ascii=False
        )




def load_history():


    history_file = (
        "results/history.json"
    )


    if not os.path.exists(history_file):

        return []


    try:

        with open(
            history_file,
            "r",
            encoding="utf-8"
        ) as f:


            return json.load(f)


    except:

        return []



# =========================
# ЗАГОЛОВОК
# =========================


st.title(
    "🛰️ Классификация спутниковых снимков"
)


st.write(
    """
    Демонстрационный модуль классификации
    спутниковых изображений.

    Выполнил: Грезин Владислав Витальевич
    Группа: УБВТ2305
    
    Используемая модель:
    EfficientNet-B0
    """
)



# =========================
# SIDEBAR СТАТИСТИКА
# =========================


st.sidebar.header(
    "Статистика модели"
)


st.sidebar.success(
    "EfficientNet-B0"
)


st.sidebar.write(
    "Accuracy: 98.52%"
)

st.sidebar.write(
    "Precision: 98.53%"
)

st.sidebar.write(
    "Recall: 98.52%"
)

st.sidebar.write(
    "F1-score: 98.51%"
)


st.sidebar.write(
    "Размер: 15.62 MB"
)

st.sidebar.write(
    "Инференс: 36.82 ms"
)



st.sidebar.divider()


history = load_history()


st.sidebar.subheader(
    "История работы"
)


st.sidebar.write(
    f"Всего распознаваний: {len(history)}"
)



if len(history) > 0:


    last = history[-1]


    st.sidebar.write(
        f"Последний класс: {last['class']}"
    )


    st.sidebar.write(
        f"Уверенность: {last['confidence']}%"
    )


    st.sidebar.write(
        f"Дата: {last['time']}"
    )




# =========================
# ЗАГРУЗКА ФАЙЛА
# =========================


uploaded_file = st.file_uploader(
    "Загрузите изображение",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)



if uploaded_file:


    image = Image.open(
        uploaded_file
    ).convert("RGB")



    st.image(
        image,
        caption="Исходное изображение",
        width=400
    )



    img = transform(image)


    img = img.unsqueeze(0)



    with torch.no_grad():


        output = model(img)


        probability = torch.softmax(
            output,
            dim=1
        )


        confidence, predicted = torch.max(
            probability,
            1
        )



    class_name = CLASSES[
        predicted.item()
    ]


    conf = (
        confidence.item()
        * 100
    )



    st.success(
        f"Класс: {class_name}"
    )


    st.info(
        f"Уверенность: {conf:.2f}%"
    )



    save_result(
        uploaded_file.name,
        class_name,
        conf
    )



# =========================
# ТАБЛИЦА HISTORY
# =========================


st.divider()


st.header(
    "История распознаваний"
)


history = load_history()



if len(history) > 0:


    df = pd.DataFrame(
        history
    )


    st.dataframe(
        df,
        use_container_width=True
    )


else:


    st.write(
        "История пока пустая"
    )