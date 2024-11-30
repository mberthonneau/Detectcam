@echo off
setlocal EnableDelayedExpansion

REM Vérifier si Python est installé
echo Verification de l'installation Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERREUR : Python n'est pas installe ou non configure dans le PATH.
    exit /b 1
)

REM Créer un fichier temporaire pour le script de vérification Python
echo import sys > check_deps.py
echo modules_to_check = [ >> check_deps.py
echo     "tkinter", >> check_deps.py
echo     "PIL", >> check_deps.py
echo     "numpy", >> check_deps.py
echo     "cv2", >> check_deps.py
echo     "fastapi", >> check_deps.py
echo     "uvicorn", >> check_deps.py
echo     "pydantic", >> check_deps.py
echo     "pandas", >> check_deps.py
echo     "openpyxl" >> check_deps.py
echo ] >> check_deps.py
echo missing_modules = [] >> check_deps.py
echo system_modules = ["tkinter"] >> check_deps.py
echo. >> check_deps.py
echo for module in modules_to_check: >> check_deps.py
echo     try: >> check_deps.py
echo         __import__(module) >> check_deps.py
echo     except ImportError: >> check_deps.py
echo         if module in system_modules: >> check_deps.py
echo             print(f"SYSTEM_MISSING:{module}") >> check_deps.py
echo         else: >> check_deps.py
echo             print(f"PIP_MISSING:{module}") >> check_deps.py

REM Exécuter la vérification des dépendances
echo Verification des dependances systeme...
python check_deps.py > deps_output.txt
set "MISSING_SYSTEM="
set "MISSING_PIP="

for /f "tokens=1,2 delims=:" %%a in (deps_output.txt) do (
    if "%%a"=="SYSTEM_MISSING" (
        set "MISSING_SYSTEM=!MISSING_SYSTEM! %%b"
    )
    if "%%a"=="PIP_MISSING" (
        set "MISSING_PIP=!MISSING_PIP! %%b"
    )
)

REM Supprimer les fichiers temporaires
del check_deps.py
del deps_output.txt

REM Vérifier les dépendances système manquantes
if not "!MISSING_SYSTEM!"=="" (
    echo.
    echo ERREUR : Certaines dependances systeme sont manquantes :!MISSING_SYSTEM!
    echo.
    echo Pour installer les dependances manquantes :
    echo.
    echo 1. Executez l'installateur Python
    echo 2. Selectionnez "Modify"
    echo 3. Assurez-vous que toutes les options sont cochees, particulierement "tcl/tk and IDLE"
    echo 4. Terminez l'installation
    echo.
    exit /b 1
)

REM Supprimer le dossier venv existant si spécifié par l'utilisateur
if "%1"=="--reset" (
    echo Suppression de l'ancien environnement virtuel...
    rmdir /s /q venv
)

REM Vérifier si l'environnement virtuel existe, sinon le créer
if not exist venv (
    echo Creation de l'environnement virtuel...
    python -m venv venv
    if errorlevel 1 (
        echo ERREUR : Impossible de créer l'environnement virtuel.
        exit /b 1
    )
)

REM Activer l'environnement virtuel
echo Activation de l'environnement virtuel...
call .\venv\Scripts\activate
if errorlevel 1 (
    echo ERREUR : Impossible d'activer l'environnement virtuel.
    exit /b 1
)

REM Installer les dépendances
echo Installation des dépendances...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERREUR : Impossible d'installer les dépendances.
    exit /b 1
)

REM Lancer l'application principale
echo Démarrage de l'application...
python main.py
if errorlevel 1 (
    echo ERREUR : L'application a rencontré une erreur.
    exit /b 1
)

REM Désactiver l'environnement virtuel
echo Désactivation de l'environnement virtuel...
deactivate
echo Terminé avec succès.

REM Nettoyage final
pause
endlocal