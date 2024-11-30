from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple, Any
from datetime import datetime, timedelta
from threading import Lock
from queue import Queue
import numpy as np
import json
import pandas as pd
import logging
from collections import defaultdict
from detectors import YOLODetector
import concurrent.futures
import os
import cv2
from threading import Thread
from pathlib import Path

@dataclass
class ROI:
    start: Tuple[int, int]
    end: Tuple[int, int]
    id: int

    @property
    def width(self) -> int:
        return abs(self.end[0] - self.start[0])

    @property
    def height(self) -> int:
        return abs(self.end[1] - self.start[1])

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.start[0] + self.end[0]) // 2,
            (self.start[1] + self.end[1]) // 2
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start': self.start,
            'end': self.end,
            'id': self.id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ROI':
        return cls(
            start=tuple(data['start']),
            end=tuple(data['end']),
            id=data['id']
        )

    def contains_point(self, x: int, y: int) -> bool:
        x1, y1 = min(self.start[0], self.end[0]), min(self.start[1], self.end[1])
        x2, y2 = max(self.start[0], self.end[0]), max(self.start[1], self.end[1])
        return x1 <= x <= x2 and y1 <= y <= y2

    def get_scaled_coordinates(self, scale_x: float, scale_y: float) -> Tuple[int, int, int, int]:
        return (
            int(min(self.start[0], self.end[0]) * scale_x),
            int(min(self.start[1], self.end[1]) * scale_y),
            int(max(self.start[0], self.end[0]) * scale_x),
            int(max(self.start[1], self.end[1]) * scale_y)
        )

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    frame_index: int
    roi_id: int
    timestamp: datetime

    @property
    def formatted_confidence(self) -> str:
        return f"{self.confidence:.1%}"

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]

    def to_dict(self):
        return {
            'label': self.label,
            'confidence': float(self.confidence),
            'bbox': tuple(map(int, self.bbox)),
            'frame_index': int(self.frame_index),
            'roi_id': int(self.roi_id),
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Detection':
        return cls(
            label=data['label'],
            confidence=data['confidence'],
            bbox=tuple(data['bbox']),
            frame_index=data['frame_index'],
            roi_id=data['roi_id'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

class FrameCache:
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self.cache: Dict[int, np.ndarray] = {}
        self.queue = Queue()
        self.lock = Lock()
        self._total_memory = 0
        self._max_memory = 1024 * 1024 * 1024

    def add_frame(self, index: int, frame: np.ndarray) -> bool:
        with self.lock:
            frame_size = frame.nbytes
            if self._total_memory + frame_size > self._max_memory:
                self._free_memory(frame_size)
            if len(self.cache) >= self.max_size:
                self._remove_oldest()
            self.cache[index] = frame.copy()
            self.queue.put(index)
            self._total_memory += frame_size
            return True

    def get_frame(self, index: int) -> Optional[np.ndarray]:
        with self.lock:
            frame = self.cache.get(index)
            return frame.copy() if frame is not None else None

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.queue = Queue()
            self._total_memory = 0

    def _remove_oldest(self):
        if not self.queue.empty():
            old_index = self.queue.get()
            old_frame = self.cache.pop(old_index, None)
            if old_frame is not None:
                self._total_memory -= old_frame.nbytes

    def _free_memory(self, needed_size: int):
        while (self._total_memory + needed_size > self._max_memory and
               not self.queue.empty()):
            self._remove_oldest()

class DetectionManager:
    def __init__(self, min_interval_seconds: int = 5):
        self.min_interval = timedelta(seconds=min_interval_seconds)
        self.last_detections: Dict[str, Dict[int, datetime]] = defaultdict(dict)
        self.all_detections: List[Detection] = []
        self.lock = Lock()
        self.statistics_cache: Dict[str, Any] = {}
        self.stats_outdated = True

    def add_detection(self, detection: Detection) -> bool:
        with self.lock:
            key = f"{detection.label}_{detection.roi_id}"
            last_time = self.last_detections[key].get(detection.roi_id)

            if last_time is None or \
               (detection.timestamp - last_time) >= self.min_interval:
                self.last_detections[key][detection.roi_id] = detection.timestamp
                self.all_detections.append(detection)
                self.stats_outdated = True
                return True
            return False

    def get_statistics(self) -> Dict[str, Any]:
        with self.lock:
            if not self.stats_outdated and self.statistics_cache:
                return self.statistics_cache.copy()

            stats = {
                'total': len(self.all_detections),
                'by_class': defaultdict(int),
                'by_roi': defaultdict(int),
                'hourly_distribution': defaultdict(int),
                'confidence_avg': defaultdict(list),
                'area_avg': defaultdict(list)
            }

            for det in self.all_detections:
                stats['by_class'][det.label] += 1
                stats['by_roi'][det.roi_id] += 1
                stats['hourly_distribution'][det.timestamp.hour] += 1
                stats['confidence_avg'][det.label].append(det.confidence)
                stats['area_avg'][det.label].append(det.area)

            for label in stats['confidence_avg']:
                stats['confidence_avg'][label] = np.mean(stats['confidence_avg'][label])
                stats['area_avg'][label] = np.mean(stats['area_avg'][label])

            self.statistics_cache = stats
            self.stats_outdated = False
            return stats.copy()

    def export_to_csv(self, filepath: str):
        with self.lock:
            df = pd.DataFrame([det.to_dict() for det in self.all_detections])
            df.to_csv(filepath, index=False)

    def export_to_excel(self, filepath: str):
        with self.lock:
            df = pd.DataFrame([det.to_dict() for det in self.all_detections])
            stats = self.get_statistics()

            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Détections', index=False)

                stats_df = pd.DataFrame({
                    'Métrique': [
                        'Total détections',
                        *[f'Détections {c}' for c in stats['by_class']],
                        *[f'ROI {r}' for r in stats['by_roi']],
                        *[f'Confiance moyenne {c}' for c in stats['confidence_avg']]
                    ],
                    'Valeur': [
                        stats['total'],
                        *stats['by_class'].values(),
                        *stats['by_roi'].values(),
                        *[f"{v:.2%}" for v in stats['confidence_avg'].values()]
                    ]
                })
                stats_df.to_excel(writer, sheet_name='Statistiques', index=False)

                hours_df = pd.DataFrame({
                    'Heure': range(24),
                    'Nombre de détections': [stats['hourly_distribution'].get(h, 0)
                                          for h in range(24)]
                })
                hours_df.to_excel(writer, sheet_name='Distribution horaire', index=False)

    def clear(self):
        with self.lock:
            self.all_detections.clear()
            self.last_detections.clear()
            self.statistics_cache.clear()
            self.stats_outdated = True

class VideoProcessor:
    def __init__(self, config: Dict[str, Any], detector: YOLODetector,
                 detection_manager: DetectionManager, frame_cache: FrameCache):
        self.config = config
        self.detector = detector
        self.detection_manager = detection_manager
        self.frame_cache = frame_cache

        self.video_path = None
        self.processing = False
        self.current_frame = None
        self.video_info = None

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count() or 4
        )
        self.processing_lock = Lock()

    def load_video(self, video_path: str) -> Dict[str, Any]:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Impossible d'ouvrir la vidéo")

            ret, frame = cap.read()
            if not ret:
                raise ValueError("Impossible de lire la vidéo")

            self.video_info = {
                'path': video_path,
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) /
                              cap.get(cv2.CAP_PROP_FPS))
            }

            self.video_path = video_path
            self.current_frame = frame
            self.frame_cache.clear()
            self.frame_cache.add_frame(0, frame)

            logging.info(f"Vidéo chargée: {video_path} "
                        f"({self.video_info['width']}x{self.video_info['height']}, "
                        f"{self.video_info['fps']:.1f} FPS, "
                        f"{self.video_info['duration']} secondes)")

            return self.video_info

        except Exception as e:
            logging.error(f"Erreur lors du chargement de la vidéo: {str(e)}")
            raise
        finally:
            if 'cap' in locals():
                cap.release()

    def _draw_roi(self, frame: np.ndarray, roi: ROI):
        """Dessine la ROI sur l'image."""
        try:
            x1 = min(roi.start[0], roi.end[0])
            y1 = min(roi.start[1], roi.end[1])
            x2 = max(roi.start[0], roi.end[0])
            y2 = max(roi.start[1], roi.end[1])
            
            # Dessiner le rectangle de la ROI
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Bleu
            # Ajouter le numéro de la ROI
            cv2.putText(frame, f"ROI {roi.id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        except Exception as e:
            logging.error(f"Erreur lors du dessin de la ROI : {str(e)}")

    def get_current_frame(self) -> Optional[np.ndarray]:
        return self.current_frame.copy() if self.current_frame is not None else None

    def get_video_info(self) -> Optional[Dict[str, Any]]:
        return self.video_info.copy() if self.video_info is not None else None

    def get_frame_at_position(self, frame_index: int) -> Optional[np.ndarray]:
        cached_frame = self.frame_cache.get_frame(frame_index)
        if cached_frame is not None:
            return cached_frame

        if not self.video_path:
            return None

        try:
            cap = cv2.VideoCapture(self.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            if ret:
                self.frame_cache.add_frame(frame_index, frame)
                return frame
            return None

        except Exception as e:
            logging.error(f"Erreur lors de la lecture de la frame {frame_index}: {str(e)}")
            return None
        finally:
            if 'cap' in locals():
                cap.release()

    def stop_processing(self):
        logging.info("Arrêt du traitement demandé")
        self.processing = False

        self.executor.shutdown(wait=False)

        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count() or 4
        )

        self.frame_cache.clear()
        logging.info("Traitement arrêté")

    def is_processing(self) -> bool:
        return self.processing

    def reset(self):
        self.stop_processing()
        self.video_path = None
        self.current_frame = None
        self.video_info = None
        self.frame_cache.clear()
        logging.info("Processeur vidéo réinitialisé")

    def start_processing(self, video_path: str, rois: List[ROI],
                        progress_callback=None, complete_callback=None):
        if self.processing:
            logging.warning("Traitement déjà en cours")
            return

        self.processing = True

        def process_thread():
            try:
                self._process_video(rois, progress_callback)
                if complete_callback:
                    complete_callback(True, None)
            except Exception as e:
                logging.error(f"Erreur de traitement: {str(e)}")
                if complete_callback:
                    complete_callback(False, str(e))
            finally:
                self.processing = False

        Thread(target=process_thread, daemon=True).start()

    def _process_video(self, rois: List[ROI], progress_callback=None):
        if not self.video_path:
            raise ValueError("Aucune vidéo chargée")

        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        try:
            batch_size = self.config['yolo']['batch_size']
            frames_batch = []
            indices_batch = []
            frame_step = max(1, int(fps / 2))

            for frame_idx in range(0, total_frames, frame_step):
                if not self.processing:
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break

                self.current_frame = frame.copy()
                frames_batch.append(frame)
                indices_batch.append(frame_idx)

                if len(frames_batch) >= batch_size:
                    self._process_batch(frames_batch, indices_batch, rois)
                    frames_batch = []
                    indices_batch = []

                if progress_callback:
                    progress = (frame_idx / total_frames) * 100
                    progress_callback(progress)

            if frames_batch and self.processing:
                self._process_batch(frames_batch, indices_batch, rois)

        except Exception as e:
            logging.error(f"Erreur lors du traitement vidéo: {str(e)}")
            raise
        finally:
            cap.release()

    def _process_batch(self, frames: List[np.ndarray],
                      indices: List[int], rois: List[ROI]):
        futures = []
        for frame, idx in zip(frames, indices):
            future = self.executor.submit(
                self._process_single_frame, frame, idx, rois
            )
            futures.append(future)

        concurrent.futures.wait(futures)

    def _process_single_frame(self, frame: np.ndarray, frame_idx: int, rois: List[ROI]):
        try:
            timestamp = datetime.now()
            frame_with_detections = frame.copy()
            all_detections = []

            # Obtenir toutes les détections de l'image
            detections = self.detector.detect(frame)
            
            # Pour chaque ROI, vérifier les détections qui sont à l'intérieur
            for roi in rois:
                # Obtenir les coordonnées de la ROI
                x1 = min(roi.start[0], roi.end[0])
                y1 = min(roi.start[1], roi.end[1])
                x2 = max(roi.start[0], roi.end[0])
                y2 = max(roi.start[1], roi.end[1])

                for det_info in detections:
                    # Obtenir les coordonnées du centre de la boîte de détection
                    bbox_x, bbox_y, bbox_w, bbox_h = det_info['bbox']
                    center_x = bbox_x + bbox_w // 2
                    center_y = bbox_y + bbox_h // 2

                    # Vérifier si le centre de la détection est dans la ROI
                    if (x1 <= center_x <= x2) and (y1 <= center_y <= y2):
                        detection = Detection(
                            label=det_info['label'],
                            confidence=det_info['confidence'],
                            bbox=det_info['bbox'],
                            frame_index=frame_idx,
                            roi_id=roi.id,
                            timestamp=timestamp
                        )

                        if self.detection_manager.add_detection(detection):
                            all_detections.append(detection)
                            self._draw_detection(frame_with_detections, detection)
                            self._draw_roi(frame_with_detections, roi)  # Dessiner la ROI aussi

            if all_detections:
                image_path = self._save_detection_frame(
                    frame_with_detections,
                    all_detections,
                    frame_idx
                )

                if hasattr(self, 'save_callback') and self.save_callback:
                    self.save_callback(image_path)

        except Exception as e:
            logging.error(f"Erreur lors du traitement de la frame {frame_idx}: {str(e)}")
            raise

    def _save_detection_frame(self, frame: np.ndarray, detections: List[Detection], frame_idx: int) -> str:
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Créer le dossier de sortie
            if self.config['save']['create_subfolders']:
                save_dir = Path(self.config['save']['output_dir']) / datetime.now().strftime("%Y-%m-%d")
            else:
                save_dir = Path(self.config['save']['output_dir'])
                
            save_dir.mkdir(parents=True, exist_ok=True)

            # Construire le nom du fichier
            image_filename = f"detection_{timestamp}_{frame_idx:06d}.{self.config['save']['format']}"
            image_path = save_dir / image_filename

            # Sauvegarder l'image
            success = cv2.imwrite(str(image_path), frame)
            if not success:
                raise RuntimeError(f"Échec de la sauvegarde de l'image: {image_path}")

            # Sauvegarder les métadonnées
            metadata = {
                'frame_index': frame_idx,
                'timestamp': timestamp,
                'detections': [det.to_dict() for det in detections]
            }

            metadata_path = image_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            logging.info(f"Image et métadonnées sauvegardées: {image_path}")
            return str(image_path)

        except Exception as e:
            logging.error(f"Erreur lors de la sauvegarde de la frame: {str(e)}")
            raise

    def set_save_callback(self, callback):
        self.save_callback = callback
    
    def _draw_detection(self, frame: np.ndarray, detection: Detection):
        """Dessine une boîte autour de la détection."""
        try:
            x, y, w, h = detection.bbox
            label = f"{detection.label} ({detection.confidence:.2f})"
            
            # Dessiner la boîte (rectangle)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Ajouter le label (texte)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            logging.error(f"Erreur lors du dessin de la détection : {str(e)}")

class StatisticsManager:
    def __init__(self, detection_manager):
        self.detection_manager = detection_manager

    def get_statistics(self):
        return {"message": "Statistiques non implémentées"}