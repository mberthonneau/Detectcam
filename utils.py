import logging
from datetime import datetime
from pathlib import Path

def setup_logging():
    try:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / f"detectcam_{datetime.now():%Y%m%d}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
    except Exception as e:
        print(f"Erreur d'initialisation du logging: {str(e)}")
        raise
