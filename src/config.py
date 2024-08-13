# Path
from pathlib import Path
import tempfile
import sys
import os
# Check if running in a PyInstaller bundle
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # If the app is running in a PyInstaller bundle
    ROOT = Path(sys._MEIPASS)
else:
    # If the app is running in a normal Python environment
    ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / 'models'
ASSETS_DIR = ROOT / 'assets'
TEXTS_DIR = ASSETS_DIR / 'texts'
CACHE_DIR = ROOT / 'cache'
STOMATA_MODEL = MODEL_DIR / 'stomata.pt'
PORE_MODEL = MODEL_DIR/'pore.pt'
if not os.path.exists(CACHE_DIR):
    os.mkdir(CACHE_DIR)

TEMP_DIR = tempfile.mkdtemp(dir=str(CACHE_DIR))

# Supported image formats
IMG_SUFFIXS = ('.jpg', '.jpeg', '.png','.bmp','.dng','.mpo','.tif','.tiff','.webp','.pfm')

# Statistical Data
#RATIO_CANOLA = 0.00021645021645021645 
# Magnification 692x  2560*1920 
SB_LENGTH = 0.05 #mm
SB_UNIT = 'mm'
SB_IN_PIXEL = 224 #pixels



SMALL_FONT_SIZE = 10
MEDIUM_FONT_SIZE = 12
BIG_FONT_SIZE = 14
# Stat constants 
STOMATA_COUNT = 'Stomata Count'
STOMATA_DENSITY= 'Stomata Density'
STOMATA_SIZE = 'Stomata Size'
PORE_SIZE ='Pore Size'
STOMATA_ID = 'Stomata ID'
FOLDER_ID = 'Folder ID'
IMAGE_ID = 'Image ID'
#------------------------------------------------------------------------------------------------
# Disabled Version 1.0
#------------------------------------------------------------------------------------------------
AERA_DEF_V1_0 = 0.21693383235647204 # Magnification 691x  2560*1920  
RATIO_DEF_V1_0 = 0.0004201680672268908 # Magnification 691x  2560*1920 git