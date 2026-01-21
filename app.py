import streamlit as st
import rasterio
from rasterio.transform import xy
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
import json
import os
import folium
from streamlit_folium import st_folium
from folium.plugins import MousePosition

st.set_page_config(page_title="Ship Detection", layout="wide")
st.title("🚢 Обнаружение кораблей на GeoTIFF изображениях")

# Загрузка модели с выбором
@st.cache_resource
def load_model(model_type):
    try:
        if model_type == "YOLO8":
            model_path = 'ship_detection_model.pt'
            model_name = "YOLOv8"
        else:  # SLA-Net
            model_path = 'ship_detection_model_best.pt'
            model_name = "SLA-Net"

        model = YOLO(model_path)
        return model, model_name
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None

# Параметры детекции
st.sidebar.header("⚙️ Параметры")

# Выбор модели
model_type = st.sidebar.selectbox(
    "🤖 Выбор модели",
    ["YOLO8", "SLA-Net"],
    index=0,
    help="YOLO8 - базовая модель, SLA-Net - улучшенная модель с Sea-Land Aware механизмом"
)

# Загружаем модель
model, model_name = load_model(model_type)

if model is None:
    st.stop()

st.sidebar.success(f"✅ {model_name} загружена!")

# Остальные параметры
conf_threshold = st.sidebar.slider("Порог уверенности", 0.1, 0.9, 0.5, 0.05)
tile_size = st.sidebar.selectbox("Размер тайла", [320, 640, 1280], index=1)
overlap = st.sidebar.slider("Перекрытие (px)", 64, 256, 128, 64)

# Информация о модели
with st.sidebar.expander("ℹ️ О моделях"):
    st.markdown(f"""
    **Текущая модель: {model_name}**

    **YOLO8:**
    - Стандартная архитектура YOLOv8
    - Быстрая детекция
    - Хорошая точность на открытой воде

    **SLA-Net:**
    - Улучшенная архитектура
    - Sea-Land Aware механизм
    - Лучше работает у берега
    - Меньше ложных срабатываний
    """)

# Загрузка файла
uploaded_file = st.file_uploader("📁 Загрузите GeoTIFF файл", type=['tif', 'tiff'])

if uploaded_file is not None:
    # Сохраняем временно
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    st.info(f"🔄 Обрабатываем изображение с помощью {model_name}...")

    # Функция обработки
    def process_geotiff(geotiff_path, model, conf_threshold, tile_size, overlap):
        with rasterio.open(geotiff_path) as src:
            img_array = src.read()
            transform = src.transform
            crs = src.crs
            bounds = src.bounds

            # Конвертация в RGB
            if src.count == 1:
                img_rgb = np.stack([img_array[0]] * 3, axis=2)
            elif src.count >= 3:
                img_rgb = np.moveaxis(img_array[:3], 0, 2)
            else:
                raise ValueError(f"Неподдерживаемое количество каналов: {src.count}")

            # Нормализация
            img_min, img_max = img_rgb.min(), img_rgb.max()
            if img_max > img_min:
                img_rgb = ((img_rgb - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_rgb = img_rgb.astype(np.uint8)

            img_height, img_width = img_rgb.shape[:2]

            # Создаем PIL изображение
            pil_image = Image.fromarray(img_rgb)

            # Разбиваем на тайлы
            tiles = []
            stride = tile_size - overlap
            for y in range(0, img_height, stride):
                for x in range(0, img_width, stride):
                    x_end = min(x + tile_size, img_width)
                    y_end = min(y + tile_size, img_height)
                    x_start = max(0, x_end - tile_size)
                    y_start = max(0, y_end - tile_size)
                    tiles.append((x_start, y_start, x_end, y_end))

            # Детекция
            all_detections = []
            progress_bar = st.progress(0)

            for i, (x1, y1, x2, y2) in enumerate(tiles):
                tile = pil_image.crop((x1, y1, x2, y2))
                tile_path = f'tile_{i}.png'
                tile.save(tile_path)

                results = model(tile_path, conf=conf_threshold, verbose=False)
                boxes = results[0].boxes

                for box in boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    abs_x1 = xyxy[0] + x1
                    abs_y1 = xyxy[1] + y1
                    abs_x2 = xyxy[2] + x1
                    abs_y2 = xyxy[3] + y1

                    all_detections.append({
                        'bbox': [abs_x1, abs_y1, abs_x2, abs_y2],
                        'conf': box.conf[0].item(),
                        'cls': int(box.cls[0].item())
                    })

                if os.path.exists(tile_path):
                    os.remove(tile_path)

                progress_bar.progress((i + 1) / len(tiles))

            # NMS
            def compute_iou(box1, box2):
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])

                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                union = area1 + area2 - intersection

                return intersection / union if union > 0 else 0

            def nms(detections, iou_threshold=0.5):
                if len(detections) == 0:
                    return []

                detections = sorted(detections, key=lambda x: x['conf'], reverse=True)
                keep = []

                while detections:
                    best = detections.pop(0)
                    keep.append(best)
                    detections = [det for det in detections 
                                if compute_iou(best['bbox'], det['bbox']) < iou_threshold]

                return keep

            final_detections = nms(all_detections, iou_threshold=0.5)

            # Преобразуем в геокоординаты
            ships_geocoords = []
            for i, det in enumerate(final_detections):
                x1, y1, x2, y2 = det['bbox']
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                lon, lat = xy(transform, center_y, center_x)

                ships_geocoords.append({
                    'id': i + 1,
                    'latitude': lat,
                    'longitude': lon,
                    'confidence': det['conf'],
                    'bbox': [float(x1), float(y1), float(x2), float(y2)]
                })

            # Рисуем рамки
            draw = ImageDraw.Draw(pil_image)
            for i, det in enumerate(final_detections):
                x1, y1, x2, y2 = det['bbox']

                # Рамка
                draw.rectangle([x1, y1, x2, y2], outline='red', width=3)

                # Номер
                label = f"#{i+1}"
                draw.text((x1 + 5, y1 + 5), label, fill='red')

            # Сохраняем результат для Folium
            pil_image.save('temp_result.png')

            return pil_image, ships_geocoords, {
                'crs': str(crs),
                'bounds': bounds,
                'transform': transform
            }

    try:
        result_image, ships, metadata = process_geotiff(
            temp_path, model, conf_threshold, tile_size, overlap
        )

        st.success(f"✅ Обнаружено кораблей: {len(ships)} (модель: {model_name})")

        # Интерактивная карта с Folium
        st.subheader("🗺️ Интерактивная карта с геопривязкой")

        bounds = metadata['bounds']
        transform = metadata['transform']

        # Центр карты
        center_lat = (bounds.bottom + bounds.top) / 2
        center_lon = (bounds.left + bounds.right) / 2

        # Создаем карту Folium
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )

        # Добавляем растровое изображение с геопривязкой
        folium.raster_layers.ImageOverlay(
            image='temp_result.png',
            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
            opacity=0.8,
            interactive=True,
            cross_origin=False
        ).add_to(m)

        # Добавляем маркеры кораблей
        for ship in ships:
            folium.CircleMarker(
                location=[ship['latitude'], ship['longitude']],
                radius=8,
                popup=f"""
                🚢 Корабль #{ship['id']}
                📍 Lat: {ship['latitude']:.6f}°
                📍 Lon: {ship['longitude']:.6f}°
                🎯 Уверенность: {ship['confidence']:.2%}
                🤖 Модель: {model_name}
                """,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7
            ).add_to(m)

        # Добавляем плагин для отображения координат при наведении
        MousePosition(
            position='topright',
            separator=' | ',
            prefix='Координаты:',
            lat_formatter="function(num) {return L.Util.formatNum(num, 6) + '° N';}",
            lng_formatter="function(num) {return L.Util.formatNum(num, 6) + '° E';}"
        ).add_to(m)

        # Отображаем карту
        st.caption(f"💡 Наведите курсор для просмотра координат. Кликните на красные маркеры для информации о кораблях. (Модель: {model_name})")
        map_data = st_folium(m, width=1400, height=700, returned_objects=["last_clicked"])

        # При клике показываем координаты
        if map_data and map_data.get('last_clicked'):
            clicked_lat = map_data['last_clicked']['lat']
            clicked_lng = map_data['last_clicked']['lng']
            st.success(f"📍 Вы кликнули: **{clicked_lat:.6f}°, {clicked_lng:.6f}°**")

        # Информация о кораблях
        st.subheader("📊 Детальная информация")
        for ship in ships:
            st.markdown(f"""
            **🚢 Корабль #{ship['id']}**
            - 📍 Координаты: {ship['latitude']:.6f}°, {ship['longitude']:.6f}°
            - 🎯 Уверенность: {ship['confidence']:.2%}
            - 📦 Bbox: [{ship['bbox'][0]:.1f}, {ship['bbox'][1]:.1f}, {ship['bbox'][2]:.1f}, {ship['bbox'][3]:.1f}]
            """)

        # Кнопка скачивания JSON
        json_data = json.dumps({
            'model': model_name,
            'model_type': model_type,
            'metadata': {
                'crs': metadata['crs'],
                'bounds': {
                    'left': metadata['bounds'].left,
                    'bottom': metadata['bounds'].bottom,
                    'right': metadata['bounds'].right,
                    'top': metadata['bounds'].top
                }
            },
            'total_ships': len(ships),
            'ships': ships
        }, indent=2)

        st.download_button(
            label=f"💾 Скачать результаты ({model_name}) (JSON)",
            data=json_data,
            file_name=f"ship_detection_{model_type}_results.json",
            mime="application/json"
        )

        # Удаляем временные файлы
        if os.path.exists(temp_path):
            os.remove(temp_path)
        if os.path.exists('temp_result.png'):
            os.remove('temp_result.png')

    except Exception as e:
        st.error(f"❌ Ошибка: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
