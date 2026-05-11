# create_submission_zip.py
import os
import zipfile
from datetime import datetime

# Configuration
AUTHOR = "Maffouo_Dongmo_Florian"
DATE = datetime.now().strftime("%Y%m%d")
ZIP_NAME = f"TP_Deep_Learning_{AUTHOR}_{DATE}.zip"

# Dossiers et fichiers à inclure (chemins relatifs)
INCLUDES = [
    ("notebooks", "notebooks"),                     # dossier notebooks
    ("src", "src"),                                 # code source
    ("models", "models"),                           # modèles sauvegardés
    ("app", "app"),                                 # application Streamlit
    ("reports", "reports"),                         # figures et rapports
    ("datasets/bank_additional", "datasets/bank_additional"), # données (optionnel)
]

# Fichiers individuels à la racine
ROOT_FILES = [
    "run_project.py",
    "download_data.py",
    "requirements.txt",
    "README.md",
    "rapport_TP_Deep_Learning_Maffouo_Dongmo_Florian.pdf",
]

def zipdir(path, zipf, arcname_prefix=""):
    """Parcourt un dossier et l'ajoute au zip."""
    for root, dirs, files in os.walk(path):
        for file in files:
            full_path = os.path.join(root, file)
            arcname = os.path.join(arcname_prefix, os.path.relpath(full_path, path))
            zipf.write(full_path, arcname)

def main():
    print(f"📦 Création de l'archive {ZIP_NAME}...")
    with zipfile.ZipFile(ZIP_NAME, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Ajout des dossiers
        for src, arc in INCLUDES:
            if os.path.exists(src):
                zipdir(src, zipf, arcname_prefix=arc)
                print(f"  ✅ Ajouté : {src}/")
            else:
                print(f"  ⚠️ Ignoré (introuvable) : {src}")
        
        # Ajout des fichiers racine
        for file in ROOT_FILES:
            if os.path.isfile(file):
                zipf.write(file, file)
                print(f"  ✅ Ajouté : {file}")
            else:
                print(f"  ⚠️ Ignoré (introuvable) : {file}")
    
    # Taille du fichier
    size_kb = os.path.getsize(ZIP_NAME) / 1024
    print(f"\n✅ Archive créée : {ZIP_NAME} ({size_kb:.1f} Ko)")

if __name__ == "__main__":
    main()