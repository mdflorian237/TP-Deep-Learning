# setup.py - Script d'installation
import subprocess
import sys
import os

def install_packages():
    """Installe les packages requis"""
    print("=" * 60)
    print("INSTALLATION DES PACKAGES")
    print("=" * 60)
    
    packages = [
        'pandas',
        'numpy', 
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'joblib',
        'streamlit',
        'plotly',
        'openpyxl'
    ]
    
    for package in packages:
        print(f"\n📦 Installation de {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("\n✅ Tous les packages sont installés!")

def create_structure():
    """Crée la structure de dossiers"""
    print("\n" + "=" * 60)
    print("CRÉATION DE LA STRUCTURE")
    print("=" * 60)
    
    folders = [
        'datasets/bank_additional',
        'datasets/fashion_mnist', 
        'notebooks',
        'src',
        'models/classical',
        'models/neural',
        'models/scalers',
        'app/pages',
        'reports/figures'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Créé: {folder}")

if __name__ == "__main__":
    create_structure()
    install_packages()
    print("\n🎯 Installation terminée! Lancez: python download_data.py")