import streamlit as st
import rasterio
from rasterio.transform import xy
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO
import json
import os
import folium
from streamlit_folium import st_folium
from folium.plugins import MousePosition

st.set_page_config(page_title="SAR Analysis", layout="wide")

# ===== БОКОВАЯ ПАНЕЛЬ С ВЫБОРОМ РЕЖИМА =====
st.sidebar.title("🧭 Навигация")
mode = st.sidebar.radio(
    "Выберите режим:",
    ["🚢 Детекция кораблей", "❄️ Детекция треков ледоколов"],
    index=0
)

st.sidebar.markdown("---")

# ============================================
# РЕЖИМ 1: ДЕТЕКЦИЯ КОРАБЛЕЙ
# ============================================
if mode == "🚢 Детекция кораблей":
    st.title("🚢 Обнаружение кораблей на GeoTIFF изображениях")

    # Загрузка модели
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
    model_type = st.sidebar.selectbox(
        "🤖 Выбор модели",
        ["YOLO8", "SLA-Net"],
        index=0,
        help="YOLO8 - базовая модель, SLA-Net - улучшенная модель"
    )

    model, model_name = load_model(model_type)
    if model is None:
        st.stop()
    st.sidebar.success(f"✅ {model_name} загружена!")

    conf_threshold = st.sidebar.slider("Порог уверенности", 0.1, 0.9, 0.5, 0.05)
    tile_size = st.sidebar.selectbox("Размер тайла", [320, 640, 1280], index=1)
    overlap = st.sidebar.slider("Перекрытие (px)", 64, 256, 128, 64)

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
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        st.info(f"🔄 Обрабатываем изображение с помощью {model_name}...")

        def process_geotiff(geotiff_path, model, conf_threshold, tile_size, overlap):
            with rasterio.open(geotiff_path) as src:
                img_array = src.read()
                transform = src.transform
                crs = src.crs
                bounds = src.bounds

                if src.count == 1:
                    img_rgb = np.stack([img_array[0]] * 3, axis=2)
                elif src.count >= 3:
                    img_rgb = np.moveaxis(img_array[:3], 0, 2)
                else:
                    raise ValueError(f"Неподдерживаемое количество каналов: {src.count}")

                img_min, img_max = img_rgb.min(), img_rgb.max()
                if img_max > img_min:
                    img_rgb = ((img_rgb - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_rgb = img_rgb.astype(np.uint8)

                img_height, img_width = img_rgb.shape[:2]
                pil_image = Image.fromarray(img_rgb)

                tiles = []
                stride = tile_size - overlap
                for y in range(0, img_height, stride):
                    for x in range(0, img_width, stride):
                        x_end = min(x + tile_size, img_width)
                        y_end = min(y + tile_size, img_height)
                        x_start = max(0, x_end - tile_size)
                        y_start = max(0, y_end - tile_size)
                        tiles.append((x_start, y_start, x_end, y_end))

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

                draw = ImageDraw.Draw(pil_image)
                for i, det in enumerate(final_detections):
                    x1, y1, x2, y2 = det['bbox']
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    label = f"#{i+1}"
                    draw.text((x1 + 5, y1 + 5), label, fill='red')

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

            st.subheader("🗺️ Интерактивная карта с геопривязкой")
            bounds = metadata['bounds']
            center_lat = (bounds.bottom + bounds.top) / 2
            center_lon = (bounds.left + bounds.right) / 2

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=12,
                tiles='OpenStreetMap'
            )

            folium.raster_layers.ImageOverlay(
                image='temp_result.png',
                bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                opacity=0.8,
                interactive=True,
                cross_origin=False
            ).add_to(m)

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

            MousePosition(
                position='topright',
                separator=' | ',
                prefix='Координаты:',
                lat_formatter="function(num) {return L.Util.formatNum(num, 6) + '° N';}",
                lng_formatter="function(num) {return L.Util.formatNum(num, 6) + '° E';}"
            ).add_to(m)

            st.caption(f"💡 Наведите курсор для просмотра координат. (Модель: {model_name})")
            map_data = st_folium(m, width=1400, height=700, returned_objects=[])

            st.subheader("📊 Детальная информация")
            for ship in ships:
                st.markdown(f"""
**🚢 Корабль #{ship['id']}**
- 📍 Координаты: {ship['latitude']:.6f}°, {ship['longitude']:.6f}°
- 🎯 Уверенность: {ship['confidence']:.2%}
""")

            json_data = json.dumps({
                'model': model_name,
                'total_ships': len(ships),
                'ships': ships
            }, indent=2)

            st.download_button(
                label=f"💾 Скачать результаты (JSON)",
                data=json_data,
                file_name=f"ship_detection_results.json",
                mime="application/json"
            )

            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists('temp_result.png'):
                os.remove('temp_result.png')

        except Exception as e:
            st.error(f"❌ Ошибка: {str(e)}")

# ============================================
# РЕЖИМ 2: ДЕТЕКЦИЯ ТРЕКОВ ЛЕДОКОЛОВ
# ============================================
else:
    st.title("❄️ Детекция треков ледоколов")

    # Параметры детекции треков
    st.sidebar.header("⚙️ Параметры детекции")

    st.sidebar.subheader("🌱 Инициализация детектора")
    min_seed_area = st.sidebar.slider("Мин. площадь семени", 10, 500, 20, 10)
    min_component_size = st.sidebar.slider("Мин. размер компоненты", 100, 1000, 300, 50)
    max_growth_steps = st.sidebar.slider("Макс. шагов роста", 1000, 10000, 5000, 500)

    st.sidebar.subheader("🔍 Пороги детекции")
    thresh_mult_high = st.sidebar.slider("Порог семян (высокий)", 0.5, 3.0, 1.5, 0.1)
    thresh_mult_low = st.sidebar.slider("Порог роста (низкий)", 0.0, 2.0, 0.1, 0.1)
    angle_tolerance = st.sidebar.slider("Допуск по углу (°)", 30, 120, 70, 5)

    st.sidebar.subheader("📏 Размеры треков")
    min_track_length = st.sidebar.slider("Мин. длина трека", 100, 2000, 700, 50)
    max_track_width = st.sidebar.slider("Макс. ширина", 10, 100, 30, 5)
    min_aspect_ratio = st.sidebar.slider("Мин. aspect ratio", 1.0, 20.0, 5.0, 0.5)

    st.sidebar.subheader("🔗 Объединение треков")
    endpoint_merge_distance = st.sidebar.slider("Расстояние слияния концов", 5, 100, 10, 5)
    edge_threshold = st.sidebar.slider("Порог края", 5, 50, 10, 5)
    max_edge_merge_distance = st.sidebar.slider("Макс. расстояние через край", 100, 2000, 1000, 100)
    merge_through_edges = st.sidebar.checkbox("Объединять через края", value=True)

    st.sidebar.subheader("✂️ Обрезка веток")
    min_branch_length = st.sidebar.slider("Мин. длина ветки", 100, 2000, 1000, 100)
    min_final_length = st.sidebar.slider("Финальная мин. длина", 500, 3000, 1500, 100)
    prune_branches = st.sidebar.checkbox("Обрезать короткие ветки", value=False)

    with st.sidebar.expander("ℹ️ О параметрах"):
        st.markdown("""
**Мин. площадь семени** - минимальный размер начального трека

**Порог семян** - яркость для поиска начальных точек

**Порог роста** - яркость для продолжения трека

**Допуск по углу** - максимальный угол поворота трека

**Расстояние слияния** - максимальное расстояние между концами для объединения

**Мин. длина ветки** - короче этого удаляются
""")

    # Загрузка детектора с параметрами
    @st.cache_resource
    def load_detector(min_seed_area, min_component_size, max_growth_steps):
        try:
            from icebreaker_detector import IcebreakerTrackDetector
            return IcebreakerTrackDetector(
                min_seed_area=min_seed_area,
                min_component_size=min_component_size,
                max_growth_steps=max_growth_steps
            )
        except Exception as e:
            st.error(f"Ошибка загрузки детектора: {e}")
            return None

    detector = load_detector(min_seed_area, min_component_size, max_growth_steps)
    if detector is None:
        st.stop()

    st.sidebar.success("✅ Детектор загружен!")

    # Загрузка файла
    uploaded_file = st.file_uploader("📁 Загрузите GeoTIFF файл с SAR изображением", type=['tif', 'tiff'])

    if uploaded_file is not None:
        temp_path = f"temp_track_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        if st.button("🚀 Запустить детекцию треков"):
            try:
                with st.spinner("⏳ Детекция треков... Это может занять несколько минут."):
                    tracks, orig_image = detector.detect_tracks(
                        temp_path,
                        thresh_mult_high=thresh_mult_high,
                        thresh_mult_low=thresh_mult_low,
                        angle_tolerance=angle_tolerance,
                        min_track_length=min_track_length,
                        max_track_width=max_track_width,
                        min_aspect_ratio=min_aspect_ratio,
                        endpoint_merge_distance=endpoint_merge_distance,
                        edge_threshold=edge_threshold,
                        max_edge_merge_distance=max_edge_merge_distance,
                        merge_through_edges=merge_through_edges,
                        min_branch_length=min_branch_length,
                        min_final_length=min_final_length,
                        prune_branches=prune_branches
                    )

                if len(tracks) > 0:
                    st.success(f"✅ Обнаружено треков: {len(tracks)}")

                    # Получаем геоданные
                    with rasterio.open(temp_path) as src:
                        transform = src.transform
                        bounds = src.bounds
                        crs = src.crs
                        img_array = src.read()

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

                    # Создаём PIL изображение
                    pil_image = Image.fromarray(img_rgb)

                    # Рисуем треки на изображении
                    draw = ImageDraw.Draw(pil_image)

                    # Генерируем цвета для треков
                    import colorsys
                    colors = []
                    for i in range(len(tracks)):
                        hue = i / max(len(tracks), 1)
                        rgb = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
                        color = tuple(int(c * 255) for c in rgb)
                        colors.append(color)

                    # Рисуем скелеты треков
                    for track_idx, track in enumerate(tracks):
                        skeleton = track.skeleton
                        coords = np.argwhere(skeleton > 0)
                        color = colors[track_idx]

                        # Рисуем утолщённую линию трека
                        for coord in coords:
                            y, x = coord
                            draw.ellipse([x-3, y-3, x+3, y+3], fill=color, outline=color)

                    # Сохраняем изображение с треками
                    result_path = 'temp_tracks_result.png'
                    pil_image.save(result_path)

                    # Создаём интерактивную карту Folium
                    st.subheader("🗺️ Интерактивная карта с треками")

                    center_lat = (bounds.bottom + bounds.top) / 2
                    center_lon = (bounds.left + bounds.right) / 2

                    m = folium.Map(
                        location=[center_lat, center_lon],
                        zoom_start=12,
                        tiles='OpenStreetMap'
                    )

                    # Накладываем изображение с треками
                    folium.raster_layers.ImageOverlay(
                        image=result_path,
                        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                        opacity=0.8,
                        interactive=True,
                        cross_origin=False
                    ).add_to(m)

                    # MousePosition
                    MousePosition(
                        position='topright',
                        separator=' | ',
                        prefix='Координаты:',
                        lat_formatter="function(num) {return L.Util.formatNum(num, 6) + '° N';}",
                        lng_formatter="function(num) {return L.Util.formatNum(num, 6) + '° E';}"
                    ).add_to(m)

                    st.caption(f"💡 Обнаружено {len(tracks)} треков. Наведите курсор для просмотра координат.")
                    map_data = st_folium(m, width=1400, height=700, returned_objects=[])

                    # Статистика
                    st.subheader("📈 Статистика треков")

                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("Всего треков", len(tracks))
                    with cols[1]:
                        total_length = sum(t.total_length for t in tracks)
                        st.metric("Общая длина", f"{total_length:.0f} пикс")
                    with cols[2]:
                        avg_length = total_length / len(tracks)
                        st.metric("Средняя длина", f"{avg_length:.0f} пикс")

                    # Детальная таблица
                    with st.expander("📊 Детальная информация по трекам"):
                        for idx, track in enumerate(tracks):
                            st.markdown(f"""
**Трек {idx+1}:**
- Длина: {track.total_length:.1f} пикселей
- Средняя ширина: {track.avg_width:.2f} пикселей
- Площадь: {track.area} пикселей
- Aspect ratio: {track.total_length/max(track.avg_width,1):.1f}
{f"- Объединён из треков: {track.merged_from}" if len(track.merged_from) > 0 else ""}
""")

                    # Экспорт результатов
                    track_data = []
                    for idx, track in enumerate(tracks):
                        try:
                            end1, end2 = track.endpoints
                            lon1, lat1 = xy(transform, end1[0], end1[1])
                            lon2, lat2 = xy(transform, end2[0], end2[1])

                            track_data.append({
                                'track_id': idx + 1,
                                'length_pixels': float(track.total_length),
                                'avg_width_pixels': float(track.avg_width),
                                'area_pixels': int(track.area),
                                'endpoint1_lat': float(lat1),
                                'endpoint1_lon': float(lon1),
                                'endpoint2_lat': float(lat2),
                                'endpoint2_lon': float(lon2),
                                'merged_from': track.merged_from
                            })
                        except:
                            track_data.append({
                                'track_id': idx + 1,
                                'length_pixels': float(track.total_length),
                                'avg_width_pixels': float(track.avg_width),
                                'area_pixels': int(track.area)
                            })

                    json_data = json.dumps({
                        'total_tracks': len(tracks),
                        'crs': str(crs),
                        'bounds': {
                            'left': bounds.left,
                            'bottom': bounds.bottom,
                            'right': bounds.right,
                            'top': bounds.top
                        },
                        'tracks': track_data
                    }, indent=2)

                    st.download_button(
                        label="💾 Скачать результаты (JSON)",
                        data=json_data,
                        file_name="icebreaker_tracks.json",
                        mime="application/json"
                    )

                    # Очистка временных файлов
                    if os.path.exists(result_path):
                        os.remove(result_path)

                else:
                    st.warning("⚠️ Треки не обнаружены. Попробуйте изменить параметры.")

                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                st.error(f"❌ Ошибка: {str(e)}")
                import traceback
                st.code(traceback.format_exc())