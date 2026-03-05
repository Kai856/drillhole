"""Shared configuration for Adavale Basin 3D model pipeline."""
import os
import numpy as np

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOMODEL_DIR = os.path.join(BASE_DIR, "Adavale 3D Geological Model")
VOXET_DIR = os.path.join(GEOMODEL_DIR, "GoCAD_Voxet")
MASKS_DIR = os.path.join(GEOMODEL_DIR, "Formation_Masks")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

VOXET_FILE = os.path.join(VOXET_DIR, "Model_Grid_3D_27_All_GumbaMask.vop1")

# Grid parameters from .vo header
NX, NY, NZ = 512, 634, 335
ORIGIN = np.array([170250.0, 7329750.0, 690.0])
SPACING = np.array([500.0, -500.0, -20.0])
NODATA = 0.0  # 0 = outside model domain (air/above surface)

# Lithology mapping from .isi file (integer code -> formation name)
LITHOLOGY_MAP = {
    1: "GRANI",    # Granite (basement intrusion)
    2: "GUMBA",    # Gumbardo Formation
    3: "EASTW",    # Eastwood Formation
    4: "LOG",      # Log Creek Formation
    5: "BURY",     # Bury Limestone
    6: "LISSO",    # Lissoy Formation
    7: "COOLA",    # Cooladdi Dolomite
    8: "BOREE",    # Boree Formation (Salt)
    9: "ETONV",    # Etonvale Formation
    10: "BUCKA",   # Buckabie Formation
    11: "GALILEE", # Galilee Basin
    12: "EROMANGA",# Eromanga Basin
    13: "GLEND",   # Glendower Formation
}

# Colors from .isi file (R, G, B normalized to 0-1)
LITHOLOGY_COLORS = {
    1:  (131/255, 112/255, 255/255),  # GRANI
    2:  (138/255,  70/255,  93/255),  # GUMBA
    3:  ( 68/255, 138/255,   0/255),  # EASTW
    4:  (126/255, 255/255,   0/255),  # LOG
    5:  (  0/255, 138/255, 138/255),  # BURY
    6:  (255/255, 246/255, 142/255),  # LISSO
    7:  (135/255, 205/255, 249/255),  # COOLA
    8:  (255/255, 130/255, 249/255),  # BOREE
    9:  (205/255, 133/255,   0/255),  # ETONV
    10: (205/255,  54/255,   0/255),  # BUCKA
    11: (  0/255, 177/255, 237/255),  # GALILEE
    12: (232/255, 149/255, 121/255),  # EROMANGA
    13: (205/255, 198/255, 114/255),  # GLEND
}

# Formation mask CSV files
MASK_FILES = {
    "GUMBA": "01_Gumba_Mask.csv",
    "EASTW": "02_Eastw_Mask.csv",
    "LOG":   "03_Log_Mask.csv",
    "BURY":  "04_Bury_Mask.csv",
    "LISSO": "05_Lisso_Mask.csv",
    "COOLA": "06_Coola_Mask.csv",
    "BOREE": "07_Boree_Mask.csv",
    "ETONV": "08_Etonv_Mask.csv",
    "BUCKA": "09_Bucka_Mask.csv",
    "GALILEE": "10_Galilee_Mask.csv",
    "GLEND": "12_Glendower_Mask.csv",
}

# Coordinate system
CRS = "GDA94 / MGA zone 55"
