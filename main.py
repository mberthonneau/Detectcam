import sys
from config import ConfigManager
from detectors import YOLODetector
from managers import FrameCache, DetectionManager, VideoProcessor, StatisticsManager
from ui import DetectionUI
from utils import setup_logging
import logging
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk
import os
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Dict, Any, List
from managers import ROI
from datetime import datetime
from pathlib import Path
import json
from utils import setup_logging

class DetectCamApp:
    def __init__(self):
        setup_logging()
        self.load_config()
        
        # Vérifier et demander le répertoire de sauvegarde au premier lancement
        if not Path(self.config['save']['output_dir']).exists():
            self.ask_output_directory()
        
        # Créer l'UI en premier
        self.ui = DetectionUI(self.config)
        
        # Initialiser les composants nécessaires avant le chargement des images
        self.initialize_components()
        
        self.current_video = None
        self.rois = []
        self.roi_counter = 0
        self.drawing_roi = False
        self.roi_start = None
        
        # Charger les images existantes après l'initialisation complète de l'UI
        self.load_existing_images()
        
        # Finir la configuration de l'UI
        self.setup_ui_callbacks()
        self.setup_canvas_bindings()
        
        logging.info("Application initialisée")

    def load_existing_images(self):
        """Charge les images existantes du répertoire de sauvegarde."""
        try:
            output_dir = Path(self.config['save']['output_dir'])
            if not output_dir.exists():
                return

            # Rechercher récursivement dans le dossier et ses sous-dossiers
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                image_files.extend(list(output_dir.rglob(ext)))
            
            # Trier par date de modification (plus récent en premier)
            image_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

            # Les ajouter à l'interface
            for image_path in image_files:
                self.ui.update_detections_list(str(image_path))

            if image_files:
                logging.info(f"{len(image_files)} images existantes chargées")
                
            # Mettre à jour le titre de la frame avec le nombre d'images
            self.ui.image_frame.config(text=f"Images détectées ({len(image_files)})")

        except Exception as e:
            logging.error(f"Erreur lors du chargement des images existantes: {str(e)}")

    def ask_output_directory(self):
        messagebox.showinfo(
            "Configuration initiale",
            "Veuillez sélectionner le répertoire où seront enregistrées les images détectées."
        )
        directory = filedialog.askdirectory(title="Sélectionner le répertoire de sauvegarde")
        if directory:
            self.config['save']['output_dir'] = directory
            ConfigManager.save_config(self.config)
        else:
            # Si l'utilisateur annule, créer un dossier par défaut
            default_dir = Path("detections")
            default_dir.mkdir(exist_ok=True)
            self.config['save']['output_dir'] = str(default_dir)
            ConfigManager.save_config(self.config)

    def setup_logging(self):
        setup_logging()

    def load_config(self):
        self.config = ConfigManager.load_config()

    def initialize_components(self):
        try:
            self.frame_cache = FrameCache()
            self.detection_manager = DetectionManager()
            self.detector = YOLODetector(self.config['yolo'])
            self.backend_info = self.detector.initialize() or "CPU"
            self.video_processor = VideoProcessor(
                config=self.config,
                detector=self.detector,
                detection_manager=self.detection_manager,
                frame_cache=self.frame_cache
            )
            self.stats_manager = StatisticsManager(self.detection_manager)

            # Ajout des callbacks pour les menus de configuration
            self.ui.bind_menu_command("Configuration", "Paramètres de détection", self.show_detection_settings)
            self.ui.bind_menu_command("Configuration", "Paramètres de sauvegarde", self.show_save_settings)
            self.ui.bind_menu_command("Visualisation", "Statistiques", self.show_statistics_window)

        except Exception as e:
            logging.error(f"Erreur d'initialisation des composants: {str(e)}")
            raise

    def setup_ui_callbacks(self):
        self.ui.bind_menu_command("Fichier", "Ouvrir vidéo", self.on_open_video)
        self.ui.bind_menu_command("Fichier", "Sauvegarder ROIs", self.on_save_rois)
        self.ui.bind_menu_command("Fichier", "Charger ROIs", self.on_load_rois)
        self.ui.bind_menu_command("Fichier", "Exporter CSV", self.on_export_csv)
        self.ui.bind_menu_command("Fichier", "Exporter Excel", self.on_export_excel)
        self.ui.bind_menu_command("Fichier", "Quitter", self.quit)

        self.ui.start_btn.config(command=self.on_start_processing)
        self.ui.stop_btn.config(command=self.video_processor.stop_processing)
        self.ui.clear_btn.config(command=self.clear_rois)

    def setup_canvas_bindings(self):
        self.ui.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.ui.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.ui.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.ui.canvas.bind("<ButtonPress-3>", self.on_canvas_right_click)

    def show_detection_settings(self):
        settings_window = tk.Toplevel(self.ui.root)
        settings_window.title("Paramètres de détection")
        settings_window.grab_set()

        # Configuration de la fenêtre
        settings_window.geometry("600x500")
        settings_window.resizable(False, False)

        # Frame principal avec scrollbar
        main_frame = ttk.Frame(settings_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Canvas et scrollbar pour le défilement
        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Paramètres YOLO
        ttk.Label(scrollable_frame, text="Paramètres YOLO", font=("Helvetica", 10, "bold")).pack(anchor="w", pady=(0, 10))

        # Fichiers de configuration
        file_frame = ttk.LabelFrame(scrollable_frame, text="Fichiers de configuration", padding=10)
        file_frame.pack(fill="x", pady=(0, 10))

        # Config path
        ttk.Label(file_frame, text="Fichier de configuration:").pack(anchor="w")
        config_path = tk.StringVar(value=self.config['yolo']['config_path'])
        config_entry = ttk.Entry(file_frame, textvariable=config_path)
        config_entry.pack(fill="x", pady=(0, 5))

        # Weights path
        ttk.Label(file_frame, text="Fichier de poids:").pack(anchor="w")
        weights_path = tk.StringVar(value=self.config['yolo']['weights_path'])
        weights_entry = ttk.Entry(file_frame, textvariable=weights_path)
        weights_entry.pack(fill="x", pady=(0, 5))

        # Names path
        ttk.Label(file_frame, text="Fichier des classes:").pack(anchor="w")
        names_path = tk.StringVar(value=self.config['yolo']['names_path'])
        names_entry = ttk.Entry(file_frame, textvariable=names_path)
        names_entry.pack(fill="x", pady=(0, 5))

        def browse_file(entry_var, file_types):
            filename = filedialog.askopenfilename(
                filetypes=file_types
            )
            if filename:
                entry_var.set(filename)

        # Boutons de parcours
        ttk.Button(
            file_frame,
            text="Parcourir config",
            command=lambda: browse_file(config_path, [("Config files", "*.cfg")])
        ).pack(pady=2)

        ttk.Button(
            file_frame,
            text="Parcourir poids",
            command=lambda: browse_file(weights_path, [("Weight files", "*.weights")])
        ).pack(pady=2)

        ttk.Button(
            file_frame,
            text="Parcourir classes",
            command=lambda: browse_file(names_path, [("Names files", "*.names")])
        ).pack(pady=2)

        # Paramètres de détection
        detect_frame = ttk.LabelFrame(scrollable_frame, text="Paramètres de détection", padding=10)
        detect_frame.pack(fill="x", pady=(0, 10))

        # Seuil de confiance
        ttk.Label(detect_frame, text="Seuil de confiance:").pack(anchor="w")
        confidence = tk.DoubleVar(value=self.config['yolo']['confidence_threshold'])
        confidence_entry = ttk.Entry(detect_frame, textvariable=confidence)
        confidence_entry.pack(fill="x", pady=(0, 5))

        # Taille du lot
        ttk.Label(detect_frame, text="Taille du lot:").pack(anchor="w")
        batch_size = tk.IntVar(value=self.config['yolo']['batch_size'])
        batch_entry = ttk.Entry(detect_frame, textvariable=batch_size)
        batch_entry.pack(fill="x", pady=(0, 5))

        # Input size
        size_frame = ttk.Frame(detect_frame)
        size_frame.pack(fill="x", pady=(0, 5))
        ttk.Label(size_frame, text="Taille d'entrée:").pack(side="left")
        input_width = tk.IntVar(value=self.config['yolo']['input_size'][0])
        input_height = tk.IntVar(value=self.config['yolo']['input_size'][1])
        ttk.Entry(size_frame, textvariable=input_width, width=6).pack(side="left", padx=2)
        ttk.Label(size_frame, text="x").pack(side="left")
        ttk.Entry(size_frame, textvariable=input_height, width=6).pack(side="left", padx=2)

        # Classes cibles avec cases à cocher et barre de défilement
        target_frame = ttk.LabelFrame(scrollable_frame, text="Classes cibles", padding=10)
        target_frame.pack(fill="x", pady=(0, 10))

        # Frame avec scrollbar pour les checkboxes
        checkbox_frame = ttk.Frame(target_frame)
        checkbox_frame.pack(fill="both", expand=True, pady=5)

        # Canvas et scrollbar pour les checkboxes
        checkbox_canvas = tk.Canvas(checkbox_frame, height=150)
        checkbox_scrollbar = ttk.Scrollbar(checkbox_frame, orient="vertical", command=checkbox_canvas.yview)
        checkbox_inner_frame = ttk.Frame(checkbox_canvas)

        checkbox_canvas.pack(side="left", fill="both", expand=True)
        checkbox_scrollbar.pack(side="right", fill="y")

        # Configurer le canvas
        checkbox_canvas.create_window((0, 0), window=checkbox_inner_frame, anchor="nw")
        checkbox_canvas.configure(yscrollcommand=checkbox_scrollbar.set)

        # Lecture des classes disponibles
        available_classes = []
        try:
            with open(self.config['yolo']['names_path'], 'r') as f:
                available_classes = [line.strip() for line in f.readlines()]
        except Exception as e:
            logging.error(f"Erreur lors de la lecture des classes: {str(e)}")

        # Variables pour les checkboxes
        checkbox_vars = {}
        
        # Créer les checkboxes
        for idx, cls in enumerate(available_classes):
            var = tk.BooleanVar(value=cls in self.config['yolo']['target_classes'])
            checkbox_vars[cls] = var
            ttk.Checkbutton(
                checkbox_inner_frame,
                text=cls,
                variable=var
            ).grid(row=idx, column=0, sticky="w", padx=5, pady=2)

        # Mettre à jour le scrolling
        checkbox_inner_frame.update_idletasks()
        checkbox_canvas.configure(scrollregion=checkbox_canvas.bbox("all"))

        def save_settings():
            try:
                # Sauvegarder les fichiers de configuration
                self.config['yolo']['config_path'] = config_path.get()
                self.config['yolo']['weights_path'] = weights_path.get()
                self.config['yolo']['names_path'] = names_path.get()
                
                # Sauvegarder les paramètres de détection
                self.config['yolo']['confidence_threshold'] = float(confidence.get())
                self.config['yolo']['batch_size'] = int(batch_size.get())
                self.config['yolo']['input_size'] = (
                    int(input_width.get()),
                    int(input_height.get())
                )
                
                # Sauvegarder les classes cibles sélectionnées
                self.config['yolo']['target_classes'] = [
                    cls for cls, var in checkbox_vars.items() if var.get()
                ]
                
                ConfigManager.save_config(self.config)
                settings_window.destroy()
                messagebox.showinfo("Succès", "Paramètres sauvegardés")
                
                # Réinitialiser le détecteur avec les nouveaux paramètres
                self.detector = YOLODetector(self.config['yolo'])
                self.video_processor.detector = self.detector
                
            except ValueError as e:
                messagebox.showerror("Erreur", f"Valeur invalide: {str(e)}")
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")

        # Boutons de sauvegarde et d'annulation
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill="x", pady=10)
        ttk.Button(button_frame, text="Sauvegarder", command=save_settings).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Annuler", command=settings_window.destroy).pack(side="left")

        # Configuration finale du scrolling
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def show_save_settings(self):
        settings_window = tk.Toplevel(self.ui.root)
        settings_window.title("Paramètres de sauvegarde")
        settings_window.grab_set()

        # Configuration de la fenêtre
        settings_window.geometry("500x300")
        settings_window.resizable(False, False)

        # Frame principal
        main_frame = ttk.Frame(settings_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Dossier de sortie
        ttk.Label(main_frame, text="Dossier de sortie:").pack(anchor="w", pady=(0, 5))
        
        # Frame pour le dossier et le bouton parcourir
        dir_frame = ttk.Frame(main_frame)
        dir_frame.pack(fill="x", pady=(0, 10))
        
        output_dir = tk.StringVar(value=self.config['save']['output_dir'])
        output_entry = ttk.Entry(dir_frame, textvariable=output_dir)
        output_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        def browse_output():
            directory = filedialog.askdirectory()
            if directory:
                output_dir.set(directory)

        ttk.Button(dir_frame, text="Parcourir", command=browse_output).pack(side="right")

        # Options de sauvegarde
        save_detections = tk.BooleanVar(value=self.config['save']['save_detections_only'])
        ttk.Checkbutton(
            main_frame,
            text="Sauvegarder uniquement les détections",
            variable=save_detections
        ).pack(anchor="w", pady=5)

        create_subfolders = tk.BooleanVar(value=self.config['save']['create_subfolders'])
        ttk.Checkbutton(
            main_frame,
            text="Créer des sous-dossiers par date",
            variable=create_subfolders
        ).pack(anchor="w", pady=5)

        def save_settings():
            self.config['save']['output_dir'] = output_dir.get()
            self.config['save']['save_detections_only'] = save_detections.get()
            self.config['save']['create_subfolders'] = create_subfolders.get()
            ConfigManager.save_config(self.config)
            settings_window.destroy()
            messagebox.showinfo("Succès", "Paramètres sauvegardés")

        # Bouton de sauvegarde
        ttk.Button(main_frame, text="Sauvegarder", command=save_settings).pack(pady=20)

    def show_statistics_window(self):
        if not self.detection_manager.all_detections:
            messagebox.showwarning(
                "Attention",
                "Aucune détection disponible pour afficher les statistiques."
            )
            return

        stats_window = tk.Toplevel(self.ui.root)
        stats_window.title("Statistiques de détection")
        stats_window.geometry("600x400")

        # Frame principal
        main_frame = ttk.Frame(stats_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        stats = self.detection_manager.get_statistics()

        # Widget Text avec scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        text_widget = tk.Text(
            text_frame,
            wrap=tk.WORD,
            yscrollcommand=scrollbar.set,
            font=("Consolas", 10)
        )
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)

        # Formater et afficher les statistiques
        text_widget.insert(tk.END, f"Total des détections: {stats['total']}\n\n")
        
        text_widget.insert(tk.END, "Détections par classe:\n")
        for classe, count in stats['by_class'].items():
            text_widget.insert(tk.END, f"- {classe}: {count}\n")
        
        text_widget.insert(tk.END, "\nDétections par ROI:\n")
        for roi, count in stats['by_roi'].items():
            text_widget.insert(tk.END, f"- ROI {roi}: {count}\n")

        text_widget.insert(tk.END, "\nConfiance moyenne par classe:\n")
        for classe, conf in stats['confidence_avg'].items():
            text_widget.insert(tk.END, f"- {classe}: {conf:.1%}\n")

        text_widget.insert(tk.END, "\nDistribution horaire:\n")
        for hour, count in stats['hourly_distribution'].items():
            text_widget.insert(tk.END, f"- {hour:02d}h: {count} détections\n")

        text_widget.config(state=tk.DISABLED)

    def on_open_video(self):
        video_path = filedialog.askopenfilename(
            title="Ouvrir une vidéo",
            filetypes=[
                ("Fichiers vidéo", "*.mp4 *.avi *.mkv *.mov"),
                ("Tous les fichiers", "*.*")
            ]
        )

        if not video_path:
            return

        try:
            self.current_video = video_path
            video_info = self.video_processor.load_video(video_path)
            first_frame = self.video_processor.get_current_frame()
            if first_frame is None:
                raise ValueError("Impossible de récupérer la première frame")

            self.display_frame(first_frame)
            self.ui.update_video_info(f"{os.path.basename(video_path)} - "
                                    f"{video_info['width']}x{video_info['height']} - "
                                    f"{video_info['fps']:.1f} FPS")
            self.ui.enable_controls()
            logging.info(f"Vidéo chargée: {video_path}")

        except Exception as e:
            logging.error(f"Erreur lors de l'ouverture de la vidéo: {str(e)}")
            messagebox.showerror("Erreur", str(e))

    def display_frame(self, frame: np.ndarray):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_h, frame_w = frame_rgb.shape[:2]
            canvas_w = self.ui.canvas.winfo_width()
            canvas_h = self.ui.canvas.winfo_height()

            ratio = min(canvas_w / frame_w, canvas_h / frame_h)
            new_w = int(frame_w * ratio)
            new_h = int(frame_h * ratio)

            frame_resized = cv2.resize(
                frame_rgb,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)

            existing_rois = self.ui.canvas.find_withtag("roi_*")
            self.ui.canvas.delete("all")

            x = (canvas_w - new_w) // 2
            y = (canvas_h - new_h) // 2

            self.ui.canvas.create_image(x, y, anchor="nw", image=photo, tags="frame")
            self.ui.canvas.image = photo

            self.redraw_all_rois()

            self.ui.canvas.update_idletasks()

        except Exception as e:
            logging.error(f"Erreur lors de l'affichage de la frame: {str(e)}")
            raise

    def update_right_panel(self, image_path=None):
        if image_path:
            self.ui.update_detections_list(image_path)

        self.ui.root.after(100, self.update_right_panel)

    def on_start_processing(self):
        if not self.current_video:
            messagebox.showwarning("Attention", "Veuillez d'abord ouvrir une vidéo.")
            return

        if not self.rois:
            messagebox.showwarning("Attention", "Veuillez définir au moins une zone de détection.")
            return

        self.ui.disable_controls()
        self.ui.clear_detections()
        self.ui.update_backend_info(self.backend_info)

        self.video_processor.save_callback = self.on_detection_saved

        def progress_update(progress):
            status_text = f"Traitement en cours ({self.backend_info})"
            detections_count = len(self.detection_manager.all_detections)
            self.ui.root.after(0, lambda: self.ui.update_progress(progress, status_text, detections_count))

        def completion_callback(success, error):
            self.ui.root.after(0, lambda: self.on_processing_complete(success, error))

        self.video_processor.start_processing(
            self.current_video,
            self.rois,
            progress_callback=progress_update,
            complete_callback=completion_callback
        )

    def on_detection_saved(self, image_path: str):
        if image_path and os.path.exists(image_path):
            self.ui.root.after(0, lambda: self.ui.update_detections_list(image_path))

    def on_processing_complete(self, success: bool, error: Optional[str]):
        self.ui.enable_controls()
        total_detections = len(self.detection_manager.all_detections)

        if success:
            messagebox.showinfo(
                "Terminé",
                f"Traitement terminé.\n{total_detections} détections trouvées."
            )
            self.ui.update_progress(100, "Terminé", total_detections)
        else:
            messagebox.showerror(
                "Erreur",
                f"Erreur lors du traitement:\n{error}"
            )
            self.ui.update_progress(0, "Erreur", total_detections)

    def on_canvas_click(self, event):
        if not self.current_video:
            messagebox.showwarning(
                "Attention",
                "Veuillez d'abord ouvrir une vidéo."
            )
            return

        self.drawing_roi = True
        self.roi_start = (event.x, event.y)
        self.ui.canvas.delete("temp_roi")

    def on_canvas_drag(self, event):
        if self.drawing_roi:
            self.ui.canvas.delete("temp_roi")
            self.ui.canvas.create_rectangle(
                self.roi_start[0], self.roi_start[1],
                event.x, event.y,
                outline="red",
                width=2,
                dash=(5, 5),
                tags="temp_roi"
            )

    def on_canvas_release(self, event):
        if not self.drawing_roi:
            return

        self.drawing_roi = False
        self.ui.canvas.delete("temp_roi")

        # Obtenir les dimensions réelles de la vidéo
        video_frame = self.video_processor.get_current_frame()
        if video_frame is None:
            return

        real_h, real_w = video_frame.shape[:2]
        canvas_w = self.ui.canvas.winfo_width()
        canvas_h = self.ui.canvas.winfo_height()

        # Calculer le ratio et les offsets de l'image dans le canvas
        ratio = min(canvas_w / real_w, canvas_h / real_h)
        new_w = int(real_w * ratio)
        new_h = int(real_h * ratio)
        offset_x = (canvas_w - new_w) // 2
        offset_y = (canvas_h - new_h) // 2

        # Ajuster les coordonnées en tenant compte des offsets
        start_x = (self.roi_start[0] - offset_x) / ratio
        start_y = (self.roi_start[1] - offset_y) / ratio
        end_x = (event.x - offset_x) / ratio
        end_y = (event.y - offset_y) / ratio

        # Vérifier que les coordonnées sont dans les limites de l'image
        start_x = max(0, min(start_x, real_w))
        start_y = max(0, min(start_y, real_h))
        end_x = max(0, min(end_x, real_w))
        end_y = max(0, min(end_y, real_h))

        # Vérifier la taille minimale
        width = abs(end_x - start_x)
        height = abs(end_y - start_y)

        if width < 20 or height < 20:
            messagebox.showwarning(
                "Attention",
                "La ROI est trop petite. Elle doit faire au moins 20x20 pixels."
            )
            return

        self.roi_counter += 1
        roi = ROI(
            start=(int(start_x), int(start_y)),
            end=(int(end_x), int(end_y)),
            id=self.roi_counter
        )

        self.rois.append(roi)
        self.ui.update_rois(self.rois)
        self.draw_roi(roi)

    def on_canvas_right_click(self, event):
        if not self.rois:
            return

        x, y = event.x, event.y
        closest_roi = min(
            self.rois,
            key=lambda r: (
                ((r.start[0] + r.end[0]) / 2 - x) ** 2 +
                ((r.start[1] + r.end[1]) / 2 - y) ** 2
            )
        )

        distance = (
            ((closest_roi.start[0] + closest_roi.end[0]) / 2 - x) ** 2 +
            ((closest_roi.start[1] + closest_roi.end[1]) / 2 - y) ** 2
        )

        if distance <= 2500:
            self.rois.remove(closest_roi)
            self.ui.update_rois(self.rois)
            self.redraw_all_rois()

    def clear_rois(self):
        if self.rois and messagebox.askyesno(
            "Confirmation",
            "Voulez-vous vraiment effacer toutes les ROIs ?"
        ):
            self.rois.clear()
            self.roi_counter = 0
            self.ui.update_rois([])

            if self.video_processor and self.video_processor.get_current_frame() is not None:
                current_frame = self.video_processor.get_current_frame()
                self.display_frame(current_frame)
            else:
                self.ui.canvas.delete("all")

    def redraw_all_rois(self):
        self.ui.canvas.delete("roi_*")
        self.ui.canvas.delete("temp_roi")

        for roi in self.rois:
            self.draw_roi(roi)

    def draw_roi(self, roi: ROI):
        try:
            # Obtenir les dimensions de l'image
            video_frame = self.video_processor.get_current_frame()
            if video_frame is None:
                return

            real_h, real_w = video_frame.shape[:2]
            canvas_w = self.ui.canvas.winfo_width()
            canvas_h = self.ui.canvas.winfo_height()

            # Calculer le ratio et les offsets
            ratio = min(canvas_w / real_w, canvas_h / real_h)
            new_w = int(real_w * ratio)
            new_h = int(real_h * ratio)
            offset_x = (canvas_w - new_w) // 2
            offset_y = (canvas_h - new_h) // 2

            # Convertir les coordonnées de l'image vers le canvas
            canvas_start_x = int(roi.start[0] * ratio) + offset_x
            canvas_start_y = int(roi.start[1] * ratio) + offset_y
            canvas_end_x = int(roi.end[0] * ratio) + offset_x
            canvas_end_y = int(roi.end[1] * ratio) + offset_y

            self.ui.canvas.create_rectangle(
                canvas_start_x, canvas_start_y,
                canvas_end_x, canvas_end_y,
                outline="blue",
                width=2,
                tags=(f"roi_{roi.id}", "roi_*")
            )
        except Exception as e:
            logging.error(f"Erreur lors du dessin de la ROI {roi.id}: {str(e)}")

    def on_config_update(self, new_config: Dict[str, Any]):
        self.config = new_config
        ConfigManager.save_config(new_config)

        self.detector = YOLODetector(self.config['yolo'])
        self.video_processor = VideoProcessor(
            self.config,
            self.detector,
            self.detection_manager,
            self.frame_cache
        )

        logging.info("Configuration mise à jour")

    def quit(self):
        if messagebox.askokcancel(
            "Quitter",
            "Voulez-vous vraiment quitter l'application ?"
        ):
            try:
                ConfigManager.save_config(self.config)
                self.video_processor.stop_processing()
                self.frame_cache.clear()

                logging.info("Application fermée proprement")
                self.ui.root.quit()

            except Exception as e:
                logging.error(f"Erreur lors de la fermeture: {str(e)}")
            finally:
                self.ui.root.destroy()

    def run(self):
        try:
            self.ui.root.mainloop()
        except Exception as e:
            logging.error(f"Erreur fatale: {str(e)}")
            raise

    def on_save_rois(self):
        if not self.rois:
            messagebox.showwarning(
                "Attention",
                "Aucune ROI à sauvegarder."
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Fichiers JSON", "*.json")]
        )

        if file_path:
            try:
                rois_data = [roi.to_dict() for roi in self.rois]
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(rois_data, f, indent=4)
                logging.info(f"ROIs sauvegardées: {file_path}")
                messagebox.showinfo("Succès", "ROIs sauvegardées avec succès.")
            except Exception as e:
                logging.error(f"Erreur de sauvegarde des ROIs: {str(e)}")
                messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde:\n{str(e)}")

    def on_load_rois(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Fichiers JSON", "*.json")]
        )

        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    rois_data = json.load(f)
                self.rois = [ROI.from_dict(data) for data in rois_data]
                self.roi_counter = max((roi.id for roi in self.rois), default=0)
                self.ui.update_rois(self.rois)
                self.redraw_all_rois()
                logging.info(f"ROIs chargées: {file_path}")
                messagebox.showinfo("Succès", "ROIs chargées avec succès.")
            except Exception as e:
                logging.error(f"Erreur de chargement des ROIs: {str(e)}")
                messagebox.showerror("Erreur", f"Erreur lors du chargement:\n{str(e)}")

    def on_export_csv(self):
        if not self.detection_manager.all_detections:
            messagebox.showwarning(
                "Attention",
                "Aucune détection à exporter."
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("Fichiers CSV", "*.csv")]
        )

        if file_path:
            try:
                self.detection_manager.export_to_csv(file_path)
                messagebox.showinfo("Succès", "Export CSV réalisé avec succès.")
            except Exception as e:
                logging.error(f"Erreur d'export CSV: {str(e)}")
                messagebox.showerror("Erreur", f"Erreur lors de l'export:\n{str(e)}")

    def on_export_excel(self):
        if not self.detection_manager.all_detections:
            messagebox.showwarning(
                "Attention",
                "Aucune détection à exporter."
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Fichiers Excel", "*.xlsx")]
        )

        if file_path:
            try:
                self.detection_manager.export_to_excel(file_path)
                messagebox.showinfo("Succès", "Export Excel réalisé avec succès.")
            except Exception as e:
                logging.error(f"Erreur d'export Excel: {str(e)}")
                messagebox.showerror("Erreur", f"Erreur lors de l'export:\n{str(e)}")


def main():
    try:
        app = DetectCamApp()
        app.run()
    except Exception as e:
        logging.error(f"Erreur fatale: {str(e)}")
        messagebox.showerror(
            "Erreur fatale",
            f"Une erreur critique est survenue:\n{str(e)}"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()