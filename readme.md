# DetectCam v0.5

DetectCam est une application Windows de détection d'objets en temps réel utilisant YOLO (You Only Look Once). Elle analyse des vidéos et détecte différents types d'objets dans des zones définies par l'utilisateur (ROI - Regions Of Interest).

## 🚀 Fonctionnalités

- Détection d'objets en temps réel avec YOLOv4-tiny
- Définition de zones de détection personnalisées
- Sauvegarde automatique des détections avec métadonnées
- Interface graphique intuitive
- Export des résultats en CSV et Excel
- Support GPU (CUDA) expérimental
- Statistiques détaillées et visualisation des résultats

## ⚙️ Prérequis

- Windows 10 ou 11
- Python 3.10 ou supérieur
- 4 Go RAM minimum (8 Go recommandé)
- Pour l'accélération GPU (expérimental) :
  - CUDA Toolkit 11.x
  - Carte graphique NVIDIA compatible CUDA

## 📥 Installation

### Méthode 1 : Installation automatique (Recommandée)

1. Téléchargez le projet :
```bash
git clone https://github.com/votre-username/detectcam.git
cd detectcam
```

2. Lancez l'installation :
```bash
run.bat
```

Le script `run.bat` gère automatiquement :
- La création de l'environnement virtuel
- L'installation des dépendances
- La vérification de la configuration
- Le lancement de l'application

### Méthode 2 : Installation manuelle

1. Téléchargez le projet :
```bash
git clone https://github.com/votre-username/detectcam.git
cd detectcam
```

2. Créez et activez l'environnement virtuel :
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Installez les dépendances :
```bash
pip install -r requirements.txt
```

4. Téléchargez les fichiers YOLO requis dans le dossier du projet :
- [yolov4-tiny.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights)
- [yolov4-tiny.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg)
- [coco.names](https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names)

5. Lancez l'application :
```bash
python main.py
```

## 📖 Guide d'utilisation rapide

1. Démarrez l'application
2. Cliquez sur "Fichier > Ouvrir vidéo"
3. Dessinez les zones de détection (clic gauche et glisser)
4. Ajustez les paramètres de détection si nécessaire
5. Cliquez sur "Lancer l'analyse"
6. Visualisez les résultats dans le panneau droit

## ⚠️ Notes importantes

- Le support CUDA est expérimental et peut ne pas fonctionner sur toutes les configurations
- L'application crée un dossier "detections" pour sauvegarder les images
- Les logs sont stockés dans le dossier "logs"
- Les ROIs peuvent être sauvegardées et rechargées

## 🔧 Paramètres configurables

### Détection
- Seuil de confiance (0 à 1)
- Taille du lot de traitement
- Classes d'objets à détecter

### Sauvegarde
- Répertoire de sortie
- Format d'image
- Organisation des dossiers par date

## 📁 Structure du projet

```
detectcam/
├── main.py           # Point d'entrée
├── config.py         # Configuration
├── detectors.py      # Détection YOLO
├── managers.py       # Gestion ROI/détections
├── ui.py            # Interface utilisateur
├── utils.py         # Utilitaires
├── server.py        # API (dev)
├── requirements.txt  # Dépendances
└── run.bat          # Script d'installation
```

## 🔍 Dépannage

### Erreurs courantes

1. "CUDA non disponible" :
   - Vérifiez l'installation de CUDA Toolkit
   - Utilisez le mode CPU (par défaut)

2. "Impossible d'ouvrir la vidéo" :
   - Vérifiez le format vidéo (MP4, AVI, MKV supportés)
   - Vérifiez les codecs installés

3. "Erreur d'installation des dépendances" :
   - Utilisez `run.bat --reset` pour réinitialiser l'environnement
   - Vérifiez votre connexion internet

## 📘 Support

Pour obtenir de l'aide :
1. Consultez les logs dans le dossier "logs"
2. Ouvrez une issue sur GitHub
3. Décrivez précisément votre problème
4. Incluez les logs pertinents

## ✨ Versions

### v0.5 (Actuelle)
- Interface graphique complète
- Support CUDA expérimental
- Export de données
- Gestion des ROI

### v0.4
- Version initiale publique
- Détection basique
- Support CPU uniquement

## 👥 Contribution

Les contributions sont bienvenues ! Pour contribuer :
1. Fork du projet
2. Création de branche (`git checkout -b feature/AmazingFeature`)
3. Commit (`git commit -m 'Add AmazingFeature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Création d'une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🤝 Contact

M. BERTHONNEAU
GitHub: [mberthonneau](https://github.com/mberthonneau)