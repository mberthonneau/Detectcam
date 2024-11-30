import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from datetime import datetime
import os
import cv2
import numpy as np
import logging
from threading import Thread
from typing import List, Optional
from detectors import YOLODetector
from managers import ROI, Detection, FrameCache, DetectionManager, VideoProcessor
from typing import Dict, Any, List, Optional
from pathlib import Path


class DetectionUI:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.callbacks = {}  # Initialisation du dictionnaire callbacks
        self.root = tk.Tk()
        self.setup_ui_components()
        
        # Appliquer les styles
        self.styles = DetectionUIStyles(self.root)
        
        # Configuration de la fenêtre principale
        self.root.title("DETECTCAM v0.5.1")
        self.root.configure(bg=self.styles.colors['background'])
    def bind_menu_command(self, menu_name: str, item_name: str, callback):
        self.callbacks[(menu_name, item_name)] = callback

    def create_menu(self):
        self.menubar = tk.Menu(self.root)

        file_menu = tk.Menu(self.menubar, tearoff=0)
        file_menu.add_command(
            label="Ouvrir vidéo",
            command=lambda: self.callbacks.get(("Fichier", "Ouvrir vidéo"), lambda: None)()
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Sauvegarder ROIs",
            command=lambda: self.callbacks.get(("Fichier", "Sauvegarder ROIs"), lambda: None)()
        )
        file_menu.add_command(
            label="Charger ROIs",
            command=lambda: self.callbacks.get(("Fichier", "Charger ROIs"), lambda: None)()
        )
        file_menu.add_separator()

        export_menu = tk.Menu(file_menu, tearoff=0)
        export_menu.add_command(
            label="Exporter CSV",
            command=lambda: self.callbacks.get(("Fichier", "Exporter CSV"), lambda: None)()
        )
        export_menu.add_command(
            label="Exporter Excel",
            command=lambda: self.callbacks.get(("Fichier", "Exporter Excel"), lambda: None)()
        )
        file_menu.add_cascade(label="Exporter", menu=export_menu)

        file_menu.add_separator()
        file_menu.add_command(
            label="Quitter",
            command=lambda: self.callbacks.get(("Fichier", "Quitter"), lambda: None)()
        )
        self.menubar.add_cascade(label="Fichier", menu=file_menu)

        config_menu = tk.Menu(self.menubar, tearoff=0)
        config_menu.add_command(
            label="Paramètres de détection",
            command=lambda: self.callbacks.get(("Configuration", "Paramètres de détection"), lambda: None)()
        )
        config_menu.add_command(
            label="Paramètres de sauvegarde",
            command=lambda: self.callbacks.get(("Configuration", "Paramètres de sauvegarde"), lambda: None)()
        )
        self.menubar.add_cascade(label="Configuration", menu=config_menu)

        viz_menu = tk.Menu(self.menubar, tearoff=0)
        viz_menu.add_command(
            label="Statistiques",
            command=lambda: self.callbacks.get(("Visualisation", "Statistiques"), lambda: None)()
        )
        self.menubar.add_cascade(label="Visualisation", menu=viz_menu)

        help_menu = tk.Menu(self.menubar, tearoff=0)
        help_menu.add_command(label="À propos", command=self.show_about)
        self.menubar.add_cascade(label="Aide", menu=help_menu)

        self.root.config(menu=self.menubar)

    def setup_controls(self):
        control_frame = ttk.Frame(self.left_frame, style='Card.TFrame')
        control_frame.pack(fill="x", padx=10, pady=5)

        # Groupe des boutons principaux
        button_group = ttk.Frame(control_frame)
        button_group.pack(fill="x", padx=5, pady=5)

        self.start_btn = ttk.Button(
            button_group,
            text="Lancer l'analyse",
            style='Primary.TButton',
            state="disabled"
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(
            button_group,
            text="Arrêter",
            style='Warning.TButton',
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)

        # Barre de progression avec étiquette
        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill="x", padx=5, pady=5)

        self.progress_label = ttk.Label(
            progress_frame,
            text="En attente...",
            style='Info.TLabel'
        )
        self.progress_label.pack(side="top", anchor="w")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            style='Custom.Horizontal.TProgressbar',
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill="x", pady=(2, 0))

    def update_video_info(self, video_name: str):
        info_text = f"Vidéo chargée : {video_name}"
        self.video_label.configure(
            text=info_text,
            style='Header.TLabel'
        )

    def enable_controls(self):
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def disable_controls(self):
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

    def show_about(self):
        messagebox.showinfo(
            "À propos",
            "DETECTCAM v0.5\.1n"
            "Application de détection d'objets dans les vidéos\n"
            "© 2024 - M. BERTHONNEAU"
        )

    def setup_ui_components(self):
        self.root.title("DETECTCAM v0.5.1")
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        window_width = int(screen_width * self.config['display']['window_width_ratio'])
        window_height = int(screen_height * self.config['display']['window_height_ratio'])
        self.root.geometry(f"{window_width}x{window_height}")

        self.create_menu()

        self.main_paned = ttk.PanedWindow(self.root, orient="horizontal")
        self.main_paned.pack(fill="both", expand=True)

        self.setup_left_panel(window_width, window_height)
        self.setup_right_panel()

    def setup_left_panel(self, window_width: int, window_height: int):
        left_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(left_frame, weight=3)

        self.video_label = ttk.Label(
            left_frame,
            text="Aucune vidéo chargée",
            wraplength=window_width * 0.6
        )
        self.video_label.pack(pady=5)

        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill="both", expand=True, pady=10)

        self.canvas = tk.Canvas(
            canvas_frame,
            width=self.config['display']['canvas_width_ratio'] * window_width,
            height=self.config['display']['canvas_height_ratio'] * window_height,
            bg="gray20"
        )
        self.canvas.pack(fill="both", expand=True)

        info_frame = ttk.Frame(left_frame)
        info_frame.pack(fill="x", padx=5, pady=5)

        self.backend_label = ttk.Label(
            info_frame,
            text="Backend: -",
            font=("Arial", 10, "bold")
        )
        self.backend_label.pack(side="left", padx=5)

        self.progress_label = ttk.Label(
            info_frame,
            text="En attente..."
        )
        self.progress_label.pack(side="right", padx=5)

        control_frame = ttk.LabelFrame(left_frame, text="Contrôles")
        control_frame.pack(fill="x", padx=5, pady=5)

        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill="x", padx=5, pady=5)

        self.start_btn = ttk.Button(
            button_frame,
            text="Lancer l'analyse",
            state="disabled"
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(
            button_frame,
            text="Arrêter",
            state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)

        self.clear_btn = ttk.Button(
            button_frame,
            text="Effacer ROIs"
        )
        self.clear_btn.pack(side="left", padx=5)

        progress_frame = ttk.Frame(control_frame)
        progress_frame.pack(fill="x", padx=5, pady=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.pack(fill="x", side="left", expand=True)

    def update_backend_info(self, backend_info: str):
        style = 'Success.TLabel' if "CUDA" in backend_info else 'Info.TLabel'
        self.backend_label.configure(
            text=f"Backend: {backend_info}",
            style=style
        )

    def setup_right_panel(self):
        right_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(right_frame, weight=1)

        roi_frame = ttk.LabelFrame(right_frame, text="Zones de détection")
        roi_frame.pack(fill="both", expand=True, padx=5, pady=5)

        roi_scroll = ttk.Scrollbar(roi_frame)
        roi_scroll.pack(side="right", fill="y")

        self.roi_listbox = tk.Listbox(
            roi_frame,
            yscrollcommand=roi_scroll.set,
            selectmode="single",
            height=10
        )
        self.roi_listbox.pack(fill="both", expand=True, padx=2)
        roi_scroll.config(command=self.roi_listbox.yview)

        self.image_frame = ttk.LabelFrame(right_frame, text="Images détectées (0)")
        self.image_frame.pack(fill="both", expand=True, padx=5, pady=5)

        det_scroll = ttk.Scrollbar(self.image_frame)
        det_scroll.pack(side="right", fill="y")

        self.image_listbox = tk.Listbox(
            self.image_frame,
            yscrollcommand=det_scroll.set,
            selectmode="single",
            height=15,
            font=("Courier", 9)
        )
        self.image_listbox.pack(fill="both", expand=True, padx=2)
        det_scroll.config(command=self.image_listbox.yview)

        preview_frame = ttk.LabelFrame(right_frame, text="Prévisualisation")
        preview_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.preview_canvas = tk.Canvas(
            preview_frame,
            width=400,
            height=400,
            bg="white"
        )
        self.preview_canvas.pack(pady=5)

        zoom_frame = ttk.Frame(preview_frame)
        zoom_frame.pack(fill="x", padx=5)

        ttk.Label(zoom_frame, text="Zoom:").pack(side="left")
        self.zoom_scale = ttk.Scale(
            zoom_frame,
            from_=0.5,
            to=3.0,
            value=1.0,
            orient="horizontal"
        )
        self.zoom_scale.pack(side="left", fill="x", expand=True)

        self.image_listbox.bind('<<ListboxSelect>>', self.on_image_select)
        self.zoom_scale.bind('<Configure>', self.on_zoom_change)

        # Ajouter le double-clic sur l'image
        self.image_listbox.bind('<Double-Button-1>', self.on_image_double_click)
        
        # Ajouter un cadre pour les boutons
        button_frame = ttk.Frame(self.image_frame)
        button_frame.pack(fill="x", padx=2, pady=2)
        
        # Ajouter le bouton pour ouvrir le répertoire
        self.open_dir_btn = ttk.Button(
            button_frame,
            text="Ouvrir le dossier des images",
            command=self.open_image_directory
        )
        self.open_dir_btn.pack(side="left", padx=2)

         # Nouveau bouton pour supprimer les images
        self.delete_images_btn = ttk.Button(
            button_frame,
            text="Supprimer toutes les images",
            #style='Warning.TButton',
            command=self.delete_all_images
        )
        self.delete_images_btn.pack(side="left", padx=2)


    def delete_all_images(self):
        try:
            # Obtenir le répertoire des images depuis la configuration
            output_dir = Path(self.config['save']['output_dir'])
            
            # Vérifier si le répertoire existe et n'est pas vide
            if not output_dir.exists() or not any(output_dir.iterdir()):
                messagebox.showinfo(
                    "Information",
                    "Aucune image à supprimer."
                )
                return
            
            # Demander confirmation
            if not messagebox.askyesno(
                "Confirmation",
                "Êtes-vous sûr de vouloir supprimer toutes les images ?\n"
                "Cette action est irréversible !"
            ):
                return
            
            # Supprimer tous les fichiers images et leurs métadonnées
            files_deleted = 0
            for file in output_dir.glob("*"):
                if file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Supprimer le fichier image
                    file.unlink()
                    
                    # Supprimer le fichier JSON associé s'il existe
                    json_file = file.with_suffix('.json')
                    if json_file.exists():
                        json_file.unlink()
                    
                    files_deleted += 1

            # Vider la liste des images dans l'interface
            self.image_listbox.delete(0, tk.END)
            self.image_frame.config(text=f"Images détectées (0)")
            self.preview_canvas.delete("all")  # Effacer la prévisualisation

            # Afficher un message de confirmation
            messagebox.showinfo(
                "Succès",
                f"{files_deleted} image(s) ont été supprimées."
            )
            
            logging.info(f"{files_deleted} images supprimées du répertoire {output_dir}")

        except Exception as e:
            logging.error(f"Erreur lors de la suppression des images: {str(e)}")
            messagebox.showerror(
                "Erreur",
                f"Une erreur est survenue lors de la suppression:\n{str(e)}"
            )

    def update_detections_list(self, image_path: str):
        try:
            if not os.path.exists(image_path):
                logging.error(f"Fichier image introuvable: {image_path}")
                return

            # Vérifier si l'image n'est pas déjà dans la liste
            items = self.image_listbox.get(0, tk.END)
            if image_path in items:
                return

            self.image_listbox.insert(0, image_path)
            self.image_listbox.selection_clear(0, tk.END)
            self.image_listbox.selection_set(0)
            self.image_listbox.see(0)

            self.on_image_select()

            count = self.image_listbox.size()
            self.image_frame.config(text=f"Images détectées ({count})")

        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour de la liste: {str(e)}")

    def on_image_select(self, event=None):
        try:
            selection = self.image_listbox.curselection()
            if not selection:
                return

            image_path = self.image_listbox.get(selection[0])
            if not os.path.exists(image_path):
                logging.error(f"Fichier image introuvable: {image_path}")
                return

            img = cv2.imread(image_path)
            if img is None:
                logging.error(f"Impossible de charger l'image: {image_path}")
                return

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)

            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()

            if canvas_width <= 1:
                canvas_width = 300
            if canvas_height <= 1:
                canvas_height = 300

            scale = self.zoom_scale.get()
            ratio = min(canvas_width/img.size[0], canvas_height/img.size[1]) * scale
            new_width = int(img.size[0] * ratio)
            new_height = int(img.size[1] * ratio)

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.preview_canvas.config(width=new_width, height=new_height)
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(
                new_width//2,
                new_height//2,
                image=photo,
                anchor="center"
            )
            self.preview_canvas.image = photo

        except Exception as e:
            logging.error(f"Erreur lors de la prévisualisation: {str(e)}")


    def on_image_double_click(self, event):
        selection = self.image_listbox.curselection()
        if not selection:
            return
            
        image_path = self.image_listbox.get(selection[0])
        if os.path.exists(image_path):
            try:
                # Utiliser l'application par défaut de Windows
                os.startfile(image_path)
            except Exception as e:
                logging.error(f"Erreur lors de l'ouverture de l'image: {str(e)}")
                messagebox.showerror("Erreur", f"Impossible d'ouvrir l'image:\n{str(e)}")

    def open_image_directory(self):
        try:
            # Obtenir le répertoire des images depuis la configuration
            output_dir = self.config['save']['output_dir']
            if not os.path.exists(output_dir):
                messagebox.showwarning(
                    "Attention",
                    "Le répertoire des images n'existe pas encore."
                )
                return
                
            # Ouvrir le répertoire avec l'explorateur Windows
            os.startfile(output_dir)
        except Exception as e:
            logging.error(f"Erreur lors de l'ouverture du répertoire: {str(e)}")
            messagebox.showerror(
                "Erreur",
                f"Impossible d'ouvrir le répertoire:\n{str(e)}"
            )

    def on_zoom_change(self, event=None):
        self.on_image_select()

    def update_rois(self, rois: List[ROI]):
        self.roi_listbox.delete(0, tk.END)
        for roi in rois:
            self.roi_listbox.insert(tk.END, f"ROI {roi.id}")

    #def update_detections_list(self, image_path: str):
    #    try:
    #        if not os.path.exists(image_path):
    #            logging.error(f"Fichier image introuvable: {image_path}")
    #            return

#            self.image_listbox.insert(0, image_path)
#            self.image_listbox.selection_clear(0, tk.END)
#            self.image_listbox.selection_set(0)
#            self.image_listbox.see(0)
#
#            self.on_image_select()
##
#           count = self.image_listbox.size()
#            self.image_frame.config(text=f"Images détectées ({count})")

#       except Exception as e:
#            logging.error(f"Erreur lors de la mise à jour de la liste: {str(e)}")

    def update_progress(self, progress: float, status_text: str = "", detections_count: int = 0):
        self.progress_var.set(progress)
        status = f"{progress:.1f}% - {status_text}"
        if detections_count > 0:
            status += f" - {detections_count} détections"
        self.progress_label.config(text=status)

    def clear_detections(self):
        self.image_listbox.delete(0, tk.END)
        self.preview_canvas.delete("all")
        self.image_frame.config(text="Images détectées (0)")

    def setup_canvas_bindings(self):
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<ButtonPress-3>", self.on_canvas_right_click)

    def on_canvas_click(self, event):
        if not self.current_video:
            messagebox.showwarning(
                "Attention",
                "Veuillez d'abord ouvrir une vidéo."
            )
            return

        self.drawing_roi = True
        self.roi_start = (event.x, event.y)
        self.canvas.delete("temp_roi")

    def on_canvas_drag(self, event):
        if self.drawing_roi:
            self.canvas.delete("temp_roi")
            self.canvas.create_rectangle(
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
        self.canvas.delete("temp_roi")

        width = abs(event.x - self.roi_start[0])
        height = abs(event.y - self.roi_start[1])

        if width < 20 or height < 20:
            messagebox.showwarning(
                "Attention",
                "La ROI est trop petite. Elle doit faire au moins 20x20 pixels."
            )
            return

        self.roi_counter += 1
        roi = ROI(
            start=self.roi_start,
            end=(event.x, event.y),
            id=self.roi_counter
        )

        self.rois.append(roi)
        self.update_rois(self.rois)
        self.draw_roi(roi)

    def on_canvas_right_click(self, event):
        if not self.rois:
            return

        x, y = event.x, event.y
        closest_roi = min(
            self.rois,
            key=lambda r: (
                (r.start[0] + r.end[0])/2 - x)**2 +
                ((r.start[1] + r.end[1])/2 - y)**2
        )

        distance = (
            ((closest_roi.start[0] + closest_roi.end[0]) / 2 - x) ** 2 +
            ((closest_roi.start[1] + closest_roi.end[1]) / 2 - y) ** 2
        )
        if distance <= 2500:
            self.rois.remove(closest_roi)
            self.update_rois(self.rois)
            self.redraw_all_rois()

    def clear_rois(self):
        if self.rois and messagebox.askyesno(
            "Confirmation",
            "Voulez-vous vraiment effacer toutes les ROIs ?"
        ):
            self.rois.clear()
            self.roi_counter = 0
            self.update_rois([])

            if self.video_processor and self.video_processor.get_current_frame() is not None:
                current_frame = self.video_processor.get_current_frame()
                self.display_frame(current_frame)
            else:
                self.canvas.delete("all")

    def redraw_all_rois(self):
        self.canvas.delete("roi_*")
        self.canvas.delete("temp_roi")

        for roi in self.rois:
            self.draw_roi(roi)

    def draw_roi(self, roi: ROI):
        try:
            self.canvas.create_rectangle(
                roi.start[0], roi.start[1],
                roi.end[0], roi.end[1],
                outline="blue",
                width=2,
                tags=(f"roi_{roi.id}", "roi_*")
            )
        except Exception as e:
            logging.error(f"Erreur lors du dessin de la ROI {roi.id}: {str(e)}")

    def display_frame(self, frame: np.ndarray):
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_h, frame_w = frame_rgb.shape[:2]
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()

            ratio = min(canvas_w/frame_w, canvas_h/frame_h)
            new_w = int(frame_w * ratio)
            new_h = int(frame_h * ratio)

            frame_resized = cv2.resize(
                frame_rgb,
                (new_w, new_h),
                interpolation=cv2.INTER_AREA
            )

            image = Image.fromarray(frame_resized)
            photo = ImageTk.PhotoImage(image)

            existing_rois = self.canvas.find_withtag("roi_*")
            self.canvas.delete("all")

            x = (canvas_w - new_w) // 2
            y = (canvas_h - new_h) // 2

            self.canvas.create_image(x, y, anchor="nw", image=photo, tags="frame")
            self.canvas.image = photo

            self.redraw_all_rois()

            self.canvas.update_idletasks()

        except Exception as e:
            logging.error(f"Erreur lors de l'affichage de la frame: {str(e)}")
            raise

class DetectionUIStyles:
    def __init__(self, root):
        self.style = ttk.Style()
        
        # Configuration des couleurs
        self.colors = {
            'primary': '#1a73e8',
            'secondary': '#5f6368',
            'success': '#1e8e3e',
            'warning': '#f9ab00',
            'error': '#d93025',
            'background': '#f8f9fa',
            'surface': '#ffffff'
        }

        # Configuration des polices
        self.fonts = {
            'header': ('Helvetica', 12, 'bold'),
            'body': ('Helvetica', 10),
            'small': ('Helvetica', 9)
        }

        # Style général de l'application
        self.style.configure('.',
            background=self.colors['background'],
            foreground=self.colors['secondary'],
            font=self.fonts['body']
        )

        # Style des boutons
        self.style.configure('Primary.TButton',
            background=self.colors['primary'],
            foreground='white',
            padding=(10, 5),
            font=self.fonts['body']
        )
        
        self.style.configure('Success.TButton',
            background=self.colors['success'],
            foreground='white',
            padding=(10, 5)
        )

        self.style.configure('Warning.TButton',
            background=self.colors['warning'],
            foreground='white',
            padding=(10, 5)
        )

        # Style des labels
        self.style.configure('Header.TLabel',
            font=self.fonts['header'],
            foreground=self.colors['primary'],
            padding=(5, 5)
        )

        self.style.configure('Info.TLabel',
            font=self.fonts['small'],
            foreground=self.colors['secondary']
        )

        # Style des frames
        self.style.configure('Card.TFrame',
            background=self.colors['surface'],
            relief='raised',
            borderwidth=1
        )

        # Style de la barre de progression
        self.style.configure('Custom.Horizontal.TProgressbar',
            background=self.colors['primary'],
            troughcolor=self.colors['background'],
            bordercolor=self.colors['background'],
            lightcolor=self.colors['primary'],
            darkcolor=self.colors['primary']
        )

        # Style des listbox (via tk)
        root.option_add('*TListbox*background', self.colors['surface'])
        root.option_add('*TListbox*foreground', self.colors['secondary'])
        root.option_add('*TListbox*selectBackground', self.colors['primary'])
        root.option_add('*TListbox*selectForeground', 'white')

    def apply_theme(self, is_dark=False):
        """Permet de basculer entre thème clair et sombre"""
        if is_dark:
            self.colors.update({
                'background': '#202124',
                'surface': '#292a2d',
                'secondary': '#e8eaed'
            })
        else:
            self.colors.update({
                'background': '#f8f9fa',
                'surface': '#ffffff',
                'secondary': '#5f6368'
            })
        # Réappliquer les styles avec les nouvelles couleurs
        self.__init__(self.root)