import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from threading import Lock

class YOLODetector:
    def __init__(self, config):
        self.config = config
        self.net = None
        self.classes = []
        self.output_layers = []
        self.cuda_enabled = False
        self.initialize()
        self.lock = Lock()

    def initialize(self):
        try:
            cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
            logging.info(f"GPUs CUDA disponibles: {cv2.cuda.getCudaEnabledDeviceCount()}")

            if cuda_available:
                try:
                    if not hasattr(cv2.dnn, 'DNN_BACKEND_CUDA'):
                        raise RuntimeError("OpenCV n'a pas été compilé avec le support CUDA")

                    self.net = cv2.dnn.readNet(
                        self.config['weights_path'],
                        self.config['config_path']
                    )

                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

                    if (self.net.getPreferableBackend() == cv2.dnn.DNN_BACKEND_CUDA and
                        self.net.getPreferableTarget() == cv2.dnn.DNN_TARGET_CUDA):

                        cv2.cuda.setDevice(0)
                        gpu_name = cv2.cuda.getDeviceName(0)
                        gpu_info = cv2.cuda.DeviceInfo()

                        cv2.cuda.setBufferPoolUsage(True)
                        cv2.cuda.setMemoryAllocator(cv2.cuda.StackAllocator())

                        self.cuda_enabled = True
                        backend_info = f"CUDA (GPU: {gpu_name})"
                        logging.info(f"CUDA activé avec succès sur {gpu_name}")

                        total_memory = gpu_info.totalMemory()
                        logging.info(f"Mémoire GPU totale: {total_memory/(1024*1024*1024):.1f}GB")
                        logging.info(f"Compute capability: {gpu_info.majorVersion()}.{gpu_info.minorVersion()}")

                    else:
                        raise RuntimeError("Échec de l'activation de CUDA")

                except Exception as cuda_error:
                    logging.error(f"Erreur détaillée CUDA: {str(cuda_error)}")
                    self._fallback_to_cpu("Erreur CUDA: " + str(cuda_error))
                    backend_info = "CPU (CUDA échec)"
            else:
                self._fallback_to_cpu("CUDA non disponible")
                backend_info = "CPU"

            with open(self.config['names_path'], "r") as f:
                self.classes = [line.strip() for line in f.readlines()]

            layer_names = self.net.getLayerNames()
            self.output_layers = [
                layer_names[i - 1]
                for i in self.net.getUnconnectedOutLayers().flatten()
            ]

            logging.info(f"YOLO initialisé avec backend: {backend_info}")
            return backend_info

        except Exception as e:
            logging.error(f"Erreur d'initialisation YOLO: {str(e)}")
            raise

    def _fallback_to_cpu(self, reason: str):
        logging.warning(f"Basculement sur CPU: {reason}")
        self.net = cv2.dnn.readNet(
            self.config['weights_path'],
            self.config['config_path']
        )
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.cuda_enabled = False

    def detect(self, frame: np.ndarray) -> list:
        with self.lock:
            try:
                height, width = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    frame,
                    1/255.0,
                    self.config['input_size'],
                    swapRB=True,
                    crop=False
                )
                self.net.setInput(blob)
                outputs = self.net.forward(self.output_layers)

                detections = []
                confidence_threshold = self.config['confidence_threshold']
                target_classes = set(self.config['target_classes'])

                for output in outputs:
                    for detection in output:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > confidence_threshold:
                            label = self.classes[class_id]
                            if label in target_classes:
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)
                                x = center_x - w//2
                                y = center_y - h//2

                                detections.append({
                                    'label': label,
                                    'confidence': float(confidence),
                                    'bbox': (x, y, w, h)
                                })

                return detections

            except Exception as e:
                logging.error(f"Erreur lors de la détection: {str(e)}")
                raise
            finally:
                blob = None
                if self.cuda_enabled:
                    cv2.cuda.streamSync()
