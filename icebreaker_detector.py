import numpy as np
import rasterio
from scipy.ndimage import uniform_filter, label, convolve
from scipy.spatial.distance import cdist
from skimage import morphology
from skimage.filters import median
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Literal
from tqdm import tqdm
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class Track:
    """Оптимизированная структура данных трека"""
    mask: np.ndarray
    seed_area: int
    grown_pixels: int
    edge_connection_length: float = 0.0
    merged_from: List[int] = field(default_factory=list)
    _skeleton: Optional[np.ndarray] = field(default=None, repr=False)
    _skeleton_length: Optional[float] = field(default=None, repr=False)
    _bbox: Optional[Tuple[int, int, int, int]] = field(default=None, repr=False)
    _endpoints: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, repr=False)

    @property
    def skeleton(self) -> np.ndarray:
        """Ленивое вычисление скелета"""
        if self._skeleton is None:
            self._skeleton = skeletonize(self.mask > 0)
        return self._skeleton

    @property
    def skeleton_length(self) -> float:
        """Ленивое вычисление длины скелета"""
        if self._skeleton_length is None:
            self._skeleton_length = float(np.sum(self.skeleton))
        return self._skeleton_length

    @property
    def total_length(self) -> float:
        """Полная длина включая соединения через края"""
        return self.skeleton_length + self.edge_connection_length

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Ленивое вычисление bounding box"""
        if self._bbox is None:
            coords = np.argwhere(self.mask > 0)
            if len(coords) > 0:
                min_row, min_col = coords.min(axis=0)
                max_row, max_col = coords.max(axis=0)
                self._bbox = (min_row, min_col, max_row, max_col)
            else:
                self._bbox = (0, 0, 0, 0)
        return self._bbox

    @property
    def endpoints(self) -> Tuple[np.ndarray, np.ndarray]:
        """Ленивое вычисление конечных точек через PCA"""
        if self._endpoints is None:
            coords = np.argwhere(self.skeleton > 0)
            if len(coords) >= 2:
                centroid = coords.mean(axis=0)
                centered = coords - centroid
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                main_axis = eigenvectors[:, np.argmax(eigenvalues)]
                projections = centered @ main_axis
                idx_min = np.argmin(projections)
                idx_max = np.argmax(projections)
                self._endpoints = (coords[idx_min], coords[idx_max])
            else:
                coords = np.argwhere(self.mask > 0)
                if len(coords) >= 2:
                    self._endpoints = (coords[0], coords[-1])
                else:
                    self._endpoints = (np.array([0, 0]), np.array([0, 0]))
        return self._endpoints

    @property
    def area(self) -> int:
        """Площадь трека"""
        return int(np.sum(self.mask > 0))

    @property
    def avg_width(self) -> float:
        """Средняя ширина трека"""
        if self.skeleton_length > 0:
            return self.area / self.skeleton_length
        min_row, min_col, max_row, max_col = self.bbox
        return min(max_row - min_row + 1, max_col - min_col + 1)

    def get_thin_mask(self) -> np.ndarray:
        """Генерация тонкой маски по требованию"""
        return self.skeleton

    def invalidate_cache(self):
        """Сброс закэшированных вычислений после изменения маски"""
        self._skeleton = None
        self._skeleton_length = None
        self._bbox = None
        self._endpoints = None


class IcebreakerTrackDetector:
    """Оптимизированный детектор треков ледоколов"""

    def __init__(self,
                 min_seed_area: int = 2000,
                 min_component_size: int = 300,
                 max_growth_steps: int = 500,
                 slice_width: int = 10):
        self.min_seed_area = min_seed_area
        self.min_component_size = min_component_size
        self.max_growth_steps = max_growth_steps
        self.slice_width = slice_width

    def load_sar_image(self, filepath: str) -> Tuple[np.ndarray, rasterio.Affine]:
        """Загрузка SAR GeoTIFF"""
        print("📂 Загрузка изображения...")
        with rasterio.open(filepath) as src:
            image = src.read(1).astype(np.float32)
            transform = src.transform
            print(f"✓ Загружено: {src.shape}, диапазон [{image.min():.1f}, {image.max():.1f}] dB")
            return image, transform

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Предобработка с оптимизацией"""
        print("🔧 Предобработка...")

        # Фильтр спекла
        image = median(image, footprint=np.ones((3, 3)))

        # Локальная нормализация
        local_mean = uniform_filter(image, size=64, mode='reflect')
        image_norm = (image - local_mean) / (local_mean + 1e-6)
        image = np.clip(image_norm, -3, 3)

        print("✓ Предобработка завершена")
        return image

    def find_seed_tracks(self, image: np.ndarray, thresh_mult_high: float = 2.0) -> List[np.ndarray]:
        """Поиск начальных семян треков"""
        print("🌱 Поиск начальных семян треков...")

        local_mean = uniform_filter(image, size=128)
        local_std = np.sqrt(uniform_filter(image**2, size=128) - local_mean**2)
        thresh_high = local_mean + thresh_mult_high * local_std

        seed_mask = image > thresh_high
        seed_mask = morphology.remove_small_objects(seed_mask, min_size=self.min_component_size)

        # Объединенная морфология вместо трех отдельных операций
        seed_mask = morphology.closing(seed_mask, morphology.disk(10))
        seed_mask = morphology.closing(seed_mask, morphology.rectangle(15, 5))
        seed_mask = morphology.closing(seed_mask, morphology.rectangle(5, 15))
        seed_mask = morphology.remove_small_objects(seed_mask, min_size=self.min_component_size * 2)

        print(f"  Семена: {np.sum(seed_mask)} пикселей ({100*np.sum(seed_mask)/seed_mask.size:.3f}%)")

        labeled, num_seeds = label(seed_mask)
        print(f"✓ Найдено {num_seeds} компонент-кандидатов")

        seeds = []
        for i in range(1, num_seeds + 1):
            component_mask = (labeled == i).astype(np.uint8)
            area = np.sum(component_mask)

            if area >= self.min_seed_area:
                seeds.append(component_mask)

        print(f"✓ Семян с площадью > {self.min_seed_area} пикс: {len(seeds)}")
        return seeds

    def get_direction_from_slice(self, mask: np.ndarray, point: np.ndarray, 
                                  slice_depth: int = 20) -> Optional[np.ndarray]:
        """Вычисление направления от точки"""
        row, col = point
        coords = np.argwhere(mask > 0)

        if len(coords) < 2:
            return None

        # Векторизованное вычисление расстояний
        distances = np.linalg.norm(coords - point, axis=1)
        mask_range = (distances > 5) & (distances < slice_depth)
        nearby = coords[mask_range]

        if len(nearby) == 0:
            return None

        centroid_nearby = nearby.mean(axis=0)
        direction = point - centroid_nearby
        norm = np.linalg.norm(direction)

        return direction / norm if norm > 0 else None

    def grow_path_vectorized(self,
                             image: np.ndarray,
                             start_point: np.ndarray,
                             direction: np.ndarray,
                             thresh_low: np.ndarray,
                             visited: np.ndarray,
                             max_steps: int = 500,
                             angle_tolerance: float = 70,
                             search_radius: int = 3) -> List[Tuple[int, int]]:
        """Оптимизированное выращивание пути с векторизацией"""
        path = []
        current = start_point.astype(float)
        current_direction = direction.copy()

        # Предвычисленные смещения для поиска
        offsets = np.array([(dr, dc) for dr in range(-search_radius, search_radius + 1)
                           for dc in range(-search_radius, search_radius + 1)
                           if not (dr == 0 and dc == 0)])

        no_progress_count = 0
        max_no_progress = 30
        height, width = image.shape
        cos_tolerance = np.cos(np.radians(angle_tolerance))

        for step in range(max_steps):
            # Векторизованный поиск кандидатов
            candidates_pos = current + offsets

            # Фильтрация по границам
            valid_mask = ((candidates_pos[:, 0] >= 0) & (candidates_pos[:, 0] < height) &
                         (candidates_pos[:, 1] >= 0) & (candidates_pos[:, 1] < width))
            candidates_pos = candidates_pos[valid_mask]

            if len(candidates_pos) == 0:
                no_progress_count += 1
                if no_progress_count > max_no_progress:
                    break
                continue

            # Преобразование в целые индексы
            candidates_idx = candidates_pos.astype(int)
            rows, cols = candidates_idx[:, 0], candidates_idx[:, 1]

            # Фильтрация по visited и threshold
            not_visited = ~visited[rows, cols]
            above_threshold = image[rows, cols] > thresh_low[rows, cols]
            valid = not_visited & above_threshold

            if not np.any(valid):
                no_progress_count += 1
                if no_progress_count > max_no_progress:
                    break
                continue

            # Фильтрация кандидатов
            valid_candidates = candidates_idx[valid]
            valid_brightness = image[valid_candidates[:, 0], valid_candidates[:, 1]]

            # Векторизованное вычисление углов
            candidate_dirs = valid_candidates - current
            norms = np.linalg.norm(candidate_dirs, axis=1)
            norms[norms == 0] = 1
            candidate_dirs = candidate_dirs / norms[:, np.newaxis]

            cos_angles = candidate_dirs @ current_direction
            cos_angles = np.clip(cos_angles, -1, 1)
            angles = np.degrees(np.arccos(cos_angles))

            # Фильтрация по углу
            angle_valid = angles <= angle_tolerance
            if not np.any(angle_valid):
                no_progress_count += 1
                if no_progress_count > max_no_progress:
                    break
                continue

            # Выбор лучшего кандидата (минимальный угол, затем максимальная яркость)
            valid_angles = angles[angle_valid]
            valid_brightness_filtered = valid_brightness[angle_valid]
            valid_candidates_filtered = valid_candidates[angle_valid]
            valid_dirs_filtered = candidate_dirs[angle_valid]

            best_idx = np.lexsort((-valid_brightness_filtered, valid_angles))[0]
            best_coord = tuple(valid_candidates_filtered[best_idx])
            best_dir = valid_dirs_filtered[best_idx]

            path.append(best_coord)
            visited[best_coord[0], best_coord[1]] = True

            current = np.array(best_coord, dtype=float)
            current_direction = 0.7 * current_direction + 0.3 * best_dir
            current_direction /= np.linalg.norm(current_direction)

            no_progress_count = 0

        return path

    def merge_tracks(self, 
                    tracks: List[Track], 
                    image_shape: Tuple[int, int],
                    merge_types: List[str] = ['edges', 'endpoints', 'overlaps'],
                    edge_threshold: int = 10,
                    max_edge_distance: int = 100,
                    endpoint_distance: int = 50) -> List[Track]:
        """
        ОПТИМИЗИРОВАННЫЙ метод объединения треков
        """
        if len(tracks) <= 1:
            return tracks

        print(f"\n🔗 Объединение треков (типы: {', '.join(merge_types)})...")

        n = len(tracks)
        connections = np.zeros((n, n), dtype=bool)
        connection_info = []

        height, width = image_shape

        # ОПТИМИЗАЦИЯ 1: Предвычисляем всё один раз
        print("  ⏳ Предвычисление данных треков...")
        track_data = []
        for track in tqdm(tracks, desc="    Кэширование", disable=len(tracks)<10):
            # Принудительно вычисляем и кэшируем
            _ = track.skeleton  # вызывает вычисление
            _ = track.endpoints
            coords = np.argwhere(track.skeleton > 0)

            track_data.append({
                'skeleton_coords': coords,
                'endpoints': track.endpoints,
                'has_data': len(coords) > 0
            })

        # ОПТИМИЗАЦИЯ 2: Быстрые проверки для каждого типа объединения
        print("  ⏳ Поиск соединений...")

        # Edges: только если нужно
        if 'edges' in merge_types:
            for i in range(n):
                if not track_data[i]['has_data']:
                    continue
                coords_i = track_data[i]['skeleton_coords']
                right_i = coords_i[coords_i[:, 1] >= width - edge_threshold]
                left_i = coords_i[coords_i[:, 1] <= edge_threshold]

                for j in range(i + 1, n):
                    if not track_data[j]['has_data']:
                        continue

                    coords_j = track_data[j]['skeleton_coords']

                    # Правый край
                    if len(right_i) > 0:
                        right_j = coords_j[coords_j[:, 1] >= width - edge_threshold]
                        if len(right_j) > 0:
                            y_diff = abs(np.mean(right_i[:, 0]) - np.mean(right_j[:, 0]))
                            if y_diff <= max_edge_distance:
                                connections[i, j] = connections[j, i] = True
                                connection_info.append((i, j, 'edge', {
                                    'side': 'right', 'distance': y_diff,
                                    'points1': right_i, 'points2': right_j
                                }))
                                print(f"    Треки {i+1} ↔ {j+1} (край правый, зазор={y_diff:.0f})")
                                continue

                    # Левый край
                    if len(left_i) > 0:
                        left_j = coords_j[coords_j[:, 1] <= edge_threshold]
                        if len(left_j) > 0:
                            y_diff = abs(np.mean(left_i[:, 0]) - np.mean(left_j[:, 0]))
                            if y_diff <= max_edge_distance:
                                connections[i, j] = connections[j, i] = True
                                connection_info.append((i, j, 'edge', {
                                    'side': 'left', 'distance': y_diff,
                                    'points1': left_i, 'points2': left_j
                                }))
                                print(f"    Треки {i+1} ↔ {j+1} (край левый, зазор={y_diff:.0f})")

        # Endpoints: используем закэшированные
        if 'endpoints' in merge_types:
            for i in range(n):
                if not track_data[i]['has_data']:
                    continue
                end1_i, end2_i = track_data[i]['endpoints']

                for j in range(i + 1, n):
                    if connections[i, j]:  # уже соединены
                        continue
                    if not track_data[j]['has_data']:
                        continue

                    end1_j, end2_j = track_data[j]['endpoints']

                    # Векторизованная проверка всех комбинаций
                    distances = [
                        np.linalg.norm(end1_i - end1_j),
                        np.linalg.norm(end1_i - end2_j),
                        np.linalg.norm(end2_i - end1_j),
                        np.linalg.norm(end2_i - end2_j)
                    ]

                    min_dist = min(distances)
                    if min_dist <= endpoint_distance:
                        connections[i, j] = connections[j, i] = True
                        connection_info.append((i, j, 'endpoint', None))
                        print(f"    Треки {i+1} ↔ {j+1} (концы, d={min_dist:.0f})")

        # Overlaps: самая медленная, делаем последней
        if 'overlaps' in merge_types:
            for i in range(n):
                for j in range(i + 1, n):
                    if connections[i, j]:  # уже соединены
                        continue

                    # ОПТИМИЗАЦИЯ: Сначала проверяем bbox
                    bbox_i = tracks[i].bbox
                    bbox_j = tracks[j].bbox

                    # Быстрая проверка: пересекаются ли bbox?
                    if not self._bboxes_overlap(bbox_i, bbox_j):
                        continue

                    # Только если bbox пересекаются - проверяем маски
                    intersection = np.logical_and(tracks[i].mask > 0, tracks[j].mask > 0)
                    if np.any(intersection):
                        connections[i, j] = connections[j, i] = True
                        connection_info.append((i, j, 'overlap', None))
                        print(f"    Треки {i+1} ↔ {j+1} (пересечение)")

        # Поиск связных компонент
        merged_groups = self._find_connected_components(connections)

        print(f"  Найдено {len(merged_groups)} групп треков")
        merged_tracks = []

        for group_idx, group in enumerate(merged_groups):
            if len(group) == 1:
                merged_tracks.append(tracks[group[0]])
            else:
                print(f"  Объединение группы {group_idx+1}: треки {[g+1 for g in group]}")
                merged_track = self._merge_track_group(
                    [tracks[idx] for idx in group],
                    group,
                    connection_info,
                    image_shape
                )
                merged_tracks.append(merged_track)

        print(f"✓ После объединения: {len(merged_tracks)} треков")
        return merged_tracks


    def _bboxes_overlap(self, bbox1: Tuple[int, int, int, int], 
                        bbox2: Tuple[int, int, int, int]) -> bool:
        """
        Быстрая проверка пересечения bounding boxes
        bbox = (min_row, min_col, max_row, max_col)
        """
        min_row1, min_col1, max_row1, max_col1 = bbox1
        min_row2, min_col2, max_row2, max_col2 = bbox2

        # Нет пересечения если один bbox полностью левее/правее/выше/ниже другого
        if max_row1 < min_row2 or max_row2 < min_row1:
            return False
        if max_col1 < min_col2 or max_col2 < min_col1:
            return False

        return True

    def _check_edge_connection(self, track1: Track, track2: Track, 
                               width: int, edge_threshold: int, 
                               max_distance: int) -> Optional[dict]:
        """Проверка соединения через края изображения"""
        coords1 = np.argwhere(track1.skeleton > 0)
        coords2 = np.argwhere(track2.skeleton > 0)

        if len(coords1) == 0 or len(coords2) == 0:
            return None

        # Проверка правого края
        right1 = coords1[coords1[:, 1] >= width - edge_threshold]
        right2 = coords2[coords2[:, 1] >= width - edge_threshold]

        if len(right1) > 0 and len(right2) > 0:
            y_diff = abs(np.mean(right1[:, 0]) - np.mean(right2[:, 0]))
            if y_diff <= max_distance:
                return {'side': 'right', 'distance': y_diff, 
                       'points1': right1, 'points2': right2}

        # Проверка левого края
        left1 = coords1[coords1[:, 1] <= edge_threshold]
        left2 = coords2[coords2[:, 1] <= edge_threshold]

        if len(left1) > 0 and len(left2) > 0:
            y_diff = abs(np.mean(left1[:, 0]) - np.mean(left2[:, 0]))
            if y_diff <= max_distance:
                return {'side': 'left', 'distance': y_diff, 
                       'points1': left1, 'points2': left2}

        return None

    def _check_endpoint_connection(self, track1: Track, track2: Track, 
                                   threshold: int) -> bool:
        """Проверка соединения на конечных точках"""
        end1_1, end1_2 = track1.endpoints
        end2_1, end2_2 = track2.endpoints

        distances = [
            np.linalg.norm(end1_1 - end2_1),
            np.linalg.norm(end1_1 - end2_2),
            np.linalg.norm(end1_2 - end2_1),
            np.linalg.norm(end1_2 - end2_2)
        ]

        return min(distances) <= threshold

    def _find_connected_components(self, connections: np.ndarray) -> List[List[int]]:
        """Поиск связных компонент через DFS"""
        n = len(connections)
        visited = [False] * n
        groups = []

        def dfs(idx, group):
            visited[idx] = True
            group.append(idx)
            for neighbor in range(n):
                if connections[idx, neighbor] and not visited[neighbor]:
                    dfs(neighbor, group)

        for i in range(n):
            if not visited[i]:
                group = []
                dfs(i, group)
                groups.append(group)

        return groups

    def _merge_track_group(self, tracks: List[Track], group_indices: List[int],
                          connection_info: List[Tuple], 
                          image_shape: Tuple[int, int]) -> Track:
        """Объединение группы треков в один"""
        merged_mask = np.zeros(image_shape, dtype=np.uint8)
        total_seed_area = 0
        total_grown_pixels = 0
        edge_connection_length = 0.0

        # Объединение масок
        for track in tracks:
            merged_mask = np.logical_or(merged_mask, track.mask > 0)
            total_seed_area += track.seed_area
            total_grown_pixels += track.grown_pixels
            edge_connection_length += track.edge_connection_length

        # Добавление линий для соединений через края
        for i, j, conn_type, conn_data in connection_info:
            if i in group_indices and j in group_indices and conn_type == 'edge' and conn_data:
                edge_connection_length += conn_data['distance']

                # Находим ближайшие точки
                pts1, pts2 = conn_data['points1'], conn_data['points2']
                if len(pts1) > 0 and len(pts2) > 0:
                    distances = cdist(pts1, pts2)
                    min_idx = np.unravel_index(np.argmin(distances), distances.shape)
                    point1 = pts1[min_idx[0]]
                    point2 = pts2[min_idx[1]]

                    # Рисуем линию
                    line_points = self._bresenham_line(
                        int(point1[0]), int(point1[1]),
                        int(point2[0]), int(point2[1])
                    )
                    for y, x in line_points:
                        if 0 <= y < image_shape[0] and 0 <= x < image_shape[1]:
                            merged_mask[y, x] = 1

        merged_mask = merged_mask.astype(np.uint8)

        return Track(
            mask=merged_mask,
            seed_area=total_seed_area,
            grown_pixels=total_grown_pixels,
            edge_connection_length=edge_connection_length,
            merged_from=[idx + 1 for idx in group_indices]
        )

    def _bresenham_line(self, y0: int, x0: int, y1: int, x1: int) -> List[Tuple[int, int]]:
        """Алгоритм Брезенхема для рисования линии"""
        points = []
        dy = abs(y1 - y0)
        dx = abs(x1 - x0)
        sy = 1 if y0 < y1 else -1
        sx = 1 if x0 < x1 else -1
        err = dx - dy
        y, x = y0, x0

        while True:
            points.append((y, x))
            if y == y1 and x == x1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points

    def validate_and_filter_tracks(self,
                                   tracks: List[Track],
                                   min_length: int = 300,
                                   max_width: int = 30,
                                   min_aspect_ratio: float = 5.0) -> List[Track]:
        """Валидация и фильтрация треков"""
        print(f"\n🔍 Валидация треков (мин.длина={min_length}, макс.ширина={max_width})...")

        validated_tracks = []

        for track_idx, track in enumerate(tqdm(tracks, desc="  Валидация", unit="трек")):
            total_length = track.total_length
            avg_width = track.avg_width
            aspect_ratio = total_length / max(avg_width, 1)

            is_valid = (total_length >= min_length and 
                       avg_width <= max_width and 
                       aspect_ratio >= min_aspect_ratio)

            if is_valid:
                validated_tracks.append(track)
                extra_str = f" (+{track.edge_connection_length:.0f} edge)" if track.edge_connection_length > 0 else ""
                print(f"\n  Трек {track_idx + 1}: ✓ длина={total_length:.0f}{extra_str}, "
                      f"ширина={avg_width:.1f}, aspect={aspect_ratio:.1f}")
            else:
                reasons = []
                if total_length < min_length:
                    reasons.append(f"короткий ({total_length:.0f}<{min_length})")
                if avg_width > max_width:
                    reasons.append(f"широкий ({avg_width:.1f}>{max_width})")
                if aspect_ratio < min_aspect_ratio:
                    reasons.append(f"не вытянутый (aspect={aspect_ratio:.1f}<{min_aspect_ratio})")
                print(f"\n  Трек {track_idx + 1}: ✗ {', '.join(reasons)}")

        print(f"\n✓ После валидации: {len(validated_tracks)}/{len(tracks)} треков")
        return validated_tracks

    def detect_tracks(self,
                     sar_filepath: str,
                     thresh_mult_high: float = 2.0,
                     thresh_mult_low: float = 1.0,
                     angle_tolerance: float = 70,
                     min_track_length: int = 300,
                     max_track_width: int = 30,
                     min_aspect_ratio: float = 5.0,
                     endpoint_merge_distance: int = 50,
                     edge_threshold: int = 10,
                     max_edge_merge_distance: int = 100,
                     merge_through_edges: bool = True,
                     min_branch_length: int = 20,
                     prune_branches: bool = False,
                     min_final_length: int = 500) -> Tuple[List[Track], np.ndarray]:
        """
        Основной метод детекции с оптимизацией

        ПОРЯДОК ОБРАБОТКИ:
        1. Загрузка и предобработка
        2. Поиск семян
        3. Выращивание треков
        4. Объединение на концах и через края
        5. Первая проверка на длину
        6. Полное объединение пересекающихся
        7. Обрезка коротких веток (петли сохраняются)
        8. Финальная проверка на длину
        """
        print("=" * 70)
        print("  ОПТИМИЗИРОВАННАЯ ДЕТЕКЦИЯ ТРЕКОВ ЛЕДОКОЛОВ")
        print("=" * 70)
        start_total = time.time()

        # ===== ШАГ 1-3: Загрузка, предобработка, поиск семян, выращивание =====
        image, transform = self.load_sar_image(sar_filepath)
        orig_image = image.copy()
        image_shape = image.shape
        image = self.preprocess(image)

        seeds = self.find_seed_tracks(image, thresh_mult_high=thresh_mult_high)
        if len(seeds) == 0:
            print("⚠️  Семена не найдены!")
            return [], orig_image

        print("🌱 Вычисление мягкого порога для выращивания...")
        local_mean = uniform_filter(image, size=32)
        local_std = np.sqrt(uniform_filter(image**2, size=32) - local_mean**2)
        thresh_low = local_mean + thresh_mult_low * local_std

        print(f"🌿 Выращивание путей от {len(seeds)} семян...")
        visited = np.zeros_like(image, dtype=bool)
        grown_tracks = []

        for seed_idx, seed_mask in enumerate(tqdm(seeds, desc="  Выращивание", unit="семя")):
            visited[seed_mask > 0] = True

            coords = np.argwhere(seed_mask > 0)
            if len(coords) < 2:
                continue

            # Находим конечные точки через PCA
            centroid = coords.mean(axis=0)
            centered = coords - centroid
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            main_axis = eigenvectors[:, np.argmax(eigenvalues)]
            projections = centered @ main_axis
            end1 = coords[np.argmin(projections)]
            end2 = coords[np.argmax(projections)]

            full_mask = seed_mask.copy()
            total_grown = 0

            # Выращивание от обоих концов
            for end_point in [end1, end2]:
                direction = self.get_direction_from_slice(seed_mask, end_point, slice_depth=20)
                if direction is not None:
                    path = self.grow_path_vectorized(
                        image, end_point, direction, thresh_low, visited,
                        max_steps=self.max_growth_steps,
                        angle_tolerance=angle_tolerance
                    )
                    total_grown += len(path)
                    for r, c in path:
                        full_mask[r, c] = 1

            # Утолщение маски
            full_mask_thick = morphology.dilation(full_mask, morphology.disk(2))

            grown_tracks.append(Track(
                mask=full_mask_thick,
                seed_area=int(np.sum(seed_mask)),
                grown_pixels=total_grown,
                edge_connection_length=0.0
            ))

        print(f"✓ Выращено {len(grown_tracks)} треков")

        # ===== ШАГ 4: Объединение на концах и через края =====
        print(f"\n{'='*70}")
        print("ШАГ 4: Объединение треков на концах и через края")
        print(f"{'='*70}")

        merge_types_initial = []
        if merge_through_edges:
            merge_types_initial.append('edges')
        merge_types_initial.append('endpoints')

        if len(merge_types_initial) > 0:
            merged_initial = self.merge_tracks(
                grown_tracks,
                image_shape,
                merge_types=merge_types_initial,
                edge_threshold=edge_threshold,
                max_edge_distance=max_edge_merge_distance,
                endpoint_distance=endpoint_merge_distance
            )
        else:
            merged_initial = grown_tracks

        # ===== ШАГ 5: Первая проверка на длину =====
        print(f"\n{'='*70}")
        print(f"ШАГ 5: Первая проверка на длину (мин={min_track_length})")
        print(f"{'='*70}")

        filtered_after_merge = []
        for track_idx, track in enumerate(merged_initial):
            if track.total_length >= min_track_length:
                filtered_after_merge.append(track)
                extra = f" (+{track.edge_connection_length:.0f} edge)" if track.edge_connection_length > 0 else ""
                print(f"  Трек {track_idx + 1}: ✓ длина={track.total_length:.0f}{extra} пикс")
            else:
                print(f"  Трек {track_idx + 1}: ✗ короткий ({track.total_length:.0f}<{min_track_length})")

        print(f"\n✓ После первой фильтрации: {len(filtered_after_merge)}/{len(merged_initial)} треков")

        if len(filtered_after_merge) == 0:
            print("⚠️  После первой фильтрации не осталось треков!")
            return [], orig_image

        # ===== ШАГ 6: Полное объединение пересекающихся треков =====
        print(f"\n{'='*70}")
        print("ШАГ 6: Полное объединение пересекающихся треков")
        print(f"{'='*70}")

        fully_merged = self.merge_tracks(
            filtered_after_merge,
            image_shape,
            merge_types=['overlaps'],
            edge_threshold=edge_threshold,
            max_edge_distance=max_edge_merge_distance,
            endpoint_distance=endpoint_merge_distance
        )
        #fully_merged = filtered_after_merge


        # ===== ШАГ 7: Обрезка коротких веток (петли сохраняются) =====

        if prune_branches:
            print(f"\n{'='*70}")
            print(f"ШАГ 7: Обрезка коротких веток (мин={min_branch_length}, петли сохраняются)")
            print(f"{'='*70}")

            pruned_tracks = []
            for track_idx, track in enumerate(tqdm(fully_merged, desc="  Обрезка", unit="трек")):
              
                print(f"\n  Трек {track_idx + 1}:")
                print(f"    Исходная длина скелета: {track.skeleton_length:.0f} пикс")

                # Удаляем короткие тупиковые ветки (петли сохраняются!)
                clean_skeleton = self._prune_short_branches(track.skeleton, min_branch_length)
                clean_length = np.sum(clean_skeleton)

                print(f"    После обрезки веток: {clean_length:.0f} пикс")
                if track.skeleton_length - clean_length > 0:
                    print(f"    Удалено {track.skeleton_length - clean_length:.0f} пикс веток")
                else:
                    print(f"    Ветки не найдены")

                # Создаем новую маску на основе очищенного скелета
                clean_mask = morphology.dilation(clean_skeleton.astype(np.uint8), morphology.disk(2))

                # Создаем новый трек с обновленной маской
                pruned_track = Track(
                    mask=clean_mask,
                    seed_area=track.seed_area,
                    grown_pixels=track.grown_pixels,
                    edge_connection_length=track.edge_connection_length,
                    merged_from=track.merged_from
                )

                pruned_tracks.append(pruned_track)

            print(f"\n✓ Обработано {len(pruned_tracks)} треков")
        else:
            pruned_tracks = fully_merged

        # ===== ШАГ 8: Финальная проверка на длину =====
        print(f"\n{'='*70}")
        print(f"ШАГ 8: Финальная валидация (длина≥{min_final_length}, ширина≤{max_track_width}, aspect≥{min_aspect_ratio})")
        print(f"{'='*70}")

        final_tracks = []
        for track_idx, track in enumerate(pruned_tracks):
            total_length = track.total_length
            avg_width = track.avg_width
            aspect_ratio = total_length / max(avg_width, 1)

            is_valid = (total_length >= min_final_length and 
                      avg_width <= max_track_width and 
                      aspect_ratio >= min_aspect_ratio)

            if is_valid:
                final_tracks.append(track)
                extra = f" (+{track.edge_connection_length:.0f} edge)" if track.edge_connection_length > 0 else ""
                print(f"  Трек {track_idx + 1}: ✓ L={total_length:.0f}{extra}, "
                      f"W={avg_width:.1f}, A={aspect_ratio:.1f}")
            else:
                reasons = []
                if total_length < min_final_length:
                    reasons.append(f"L={total_length:.0f}<{min_final_length}")
                if avg_width > max_track_width:
                    reasons.append(f"W={avg_width:.1f}>{max_track_width}")
                if aspect_ratio < min_aspect_ratio:
                    reasons.append(f"A={aspect_ratio:.1f}<{min_aspect_ratio}")
                print(f"  Трек {track_idx + 1}: ✗ {', '.join(reasons)}")

        print(f"\n✓ После финальной валидации: {len(final_tracks)}/{len(pruned_tracks)} треков")

        # ===== ЗАВЕРШЕНИЕ =====
        elapsed = time.time() - start_total
        print(f"\n{'='*70}")
        print(f"✓ ДЕТЕКЦИЯ ЗАВЕРШЕНА за {elapsed:.1f}с")
        print(f"  Найдено финальных треков: {len(final_tracks)}")
        if len(final_tracks) > 0:
            total_len = sum(t.total_length for t in final_tracks)
            avg_len = total_len / len(final_tracks)
            avg_width = np.mean([t.avg_width for t in final_tracks])
            print(f"  Общая длина: {total_len:.0f} пикс")
            print(f"  Средняя длина: {avg_len:.0f} пикс")
            print(f"  Средняя ширина: {avg_width:.1f} пикс")
        print(f"{'='*70}")

        return final_tracks, orig_image


    def _prune_short_branches(self, skeleton: np.ndarray, min_branch_length: int = 20) -> np.ndarray:
        """
        ОПТИМИЗИРОВАННАЯ версия: удаляет короткие тупиковые ветки быстрее
        """
        kernel = np.array([[1, 1, 1],
                          [1, 0, 1],
                          [1, 1, 1]])

        clean_skeleton = skeleton.copy()

        # ОПТИМИЗАЦИЯ: Ограничиваем итерации и обрабатываем все ветки за раз
        max_iterations = 10  # вместо 100

        for iteration in range(max_iterations):
            # Находим конечные точки ОДИН раз за итерацию
            neighbor_count = convolve(clean_skeleton.astype(int), kernel, mode='constant')
            end_points = (clean_skeleton > 0) & (neighbor_count == 1)
            end_coords = np.argwhere(end_points > 0)

            if len(end_coords) == 0:
                break  # нет конечных точек - выходим

            # Обрабатываем все короткие ветки за одну итерацию
            branches_to_remove = []

            for end_point in end_coords:
                branch = self._trace_branch_to_junction_fast(clean_skeleton, end_point)

                if len(branch) < min_branch_length:
                    branches_to_remove.extend(branch)

            # Удаляем все найденные ветки СРАЗУ
            if len(branches_to_remove) == 0:
                break

            for r, c in branches_to_remove:
                clean_skeleton[r, c] = 0

        return clean_skeleton


    def _trace_branch_to_junction_fast(self, skeleton: np.ndarray, start: np.ndarray) -> List[Tuple[int, int]]:
        """
        БЫСТРАЯ версия трассировки: без kernel, меньше проверок
        """
        path = [tuple(start)]
        current = start.copy()
        visited = {tuple(start)}

        # Предвычисленные смещения
        offsets = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]

        for step in range(100):  # вместо 1000
            # Быстрый поиск соседей
            neighbors = []
            for dr, dc in offsets:
                nr, nc = current[0] + dr, current[1] + dc

                if (0 <= nr < skeleton.shape[0] and
                    0 <= nc < skeleton.shape[1] and
                    skeleton[nr, nc] > 0 and
                    (nr, nc) not in visited):
                    neighbors.append((nr, nc))

            if len(neighbors) == 0:
                break  # тупик

            if len(neighbors) > 1:
                break  # развилка - не включаем её

            # Единственный сосед - продолжаем
            next_point = neighbors[0]

            # БЫСТРАЯ проверка: это точка ветвления?
            # Считаем соседей напрямую, без convolve
            nr, nc = next_point
            neighbor_count = 0
            for dr, dc in offsets:
                nnr, nnc = nr + dr, nc + dc
                if (0 <= nnr < skeleton.shape[0] and
                    0 <= nnc < skeleton.shape[1] and
                    skeleton[nnr, nnc] > 0):
                    neighbor_count += 1

            if neighbor_count >= 3:
                break  # точка ветвления - останавливаемся

            path.append(next_point)
            visited.add(next_point)
            current = np.array(next_point)

        return path

    def visualize_tracks(self, 
                        tracks: List[Track], 
                        image: np.ndarray,
                        show_skeleton: bool = True,
                        figsize: Tuple[int, int] = (20, 10)):
        """Визуализация результатов"""
        print("\n🎨 Визуализация треков...")

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Исходное изображение
        axes[0].imshow(image, cmap='gray', vmin=np.percentile(image, 1), 
                      vmax=np.percentile(image, 99))
        axes[0].set_title('Исходное изображение')
        axes[0].axis('off')

        # Детектированные треки
        axes[1].imshow(image, cmap='gray', vmin=np.percentile(image, 1), 
                      vmax=np.percentile(image, 99))

        for track_idx, track in enumerate(tracks):
            if show_skeleton:
                mask_display = track.skeleton
                color = plt.cm.rainbow(track_idx / max(len(tracks), 1))
                axes[1].contour(mask_display, colors=[color], linewidths=2, levels=[0.5])
            else:
                mask_display = track.mask
                axes[1].contour(mask_display, colors=['red'], linewidths=1, levels=[0.5])

            # Подпись трека
            coords = np.argwhere(mask_display > 0)
            if len(coords) > 0:
                center = coords.mean(axis=0)
                axes[1].text(center[1], center[0], str(track_idx + 1), 
                           color='yellow', fontsize=12, fontweight='bold',
                           ha='center', va='center')
        
        axes[1].set_title(f'Детектированные треки: {len(tracks)}')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()

        # Статистика
        print("\n📊 Статистика треков:")
        print("-" * 70)
        print(f"{'№':<5} {'Длина':<12} {'Ширина':<12} {'Aspect':<10} {'Площадь':<10}")
        print("-" * 70)

        for idx, track in enumerate(tracks):
            print(f"{idx+1:<5} {track.total_length:<12.1f} {track.avg_width:<12.2f} "
                  f"{track.total_length/max(track.avg_width,1):<10.1f} {track.area:<10}")

        print("-" * 70)
        print(f"Всего треков: {len(tracks)}")
        print(f"Общая длина: {sum(t.total_length for t in tracks):.1f} пикс")
        print(f"Средняя длина: {np.mean([t.total_length for t in tracks]):.1f} пикс")
        print(f"Средняя ширина: {np.mean([t.avg_width for t in tracks]):.2f} пикс")


# Пример использования
if __name__ == "__main__":
    detector = IcebreakerTrackDetector(
        min_seed_area=20,
        min_component_size=300,
        max_growth_steps=5000
    )

    tracks, image = detector.detect_tracks(
        sar_filepath="7.tiff",
        thresh_mult_high=1.5,
        thresh_mult_low=0.1,
        angle_tolerance=70,
        min_track_length=1000,
        max_track_width=30,
        min_aspect_ratio=5.0,
        endpoint_merge_distance=10,
        edge_threshold=10,
        max_edge_merge_distance=1000,
        merge_through_edges=True,
        min_branch_length=1000,
        min_final_length=1500,
        prune_branches=False
    )

    detector.visualize_tracks(tracks, image, show_skeleton=True)