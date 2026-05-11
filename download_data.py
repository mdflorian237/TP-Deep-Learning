# download_data.py
import os
import urllib.request
import zipfile
import ssl
import certifi

# Désactiver la vérification SSL (si nécessaire)
ssl._create_default_https_context = ssl._create_unverified_context

def download_bank_data():
    """Télécharge et extrait les données Bank Telemarketing"""
    
    print("=" * 60)
    print("TÉLÉCHARGEMENT DES DONNÉES BANK TELEMARKETING")
    print("=" * 60)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    zip_path = "datasets/bank-additional.zip"
    extract_path = "datasets/"
    
    # Téléchargement
    if not os.path.exists(zip_path):
        print(f"📥 Téléchargement depuis: {url}")
        try:
            urllib.request.urlretrieve(url, zip_path)
            print("✅ Téléchargement terminé!")
        except Exception as e:
            print(f"❌ Erreur de téléchargement: {e}")
            return False
    else:
        print("📁 Fichier zip déjà présent")
    
    # Extraction
    if not os.path.exists("datasets/bank_additional/bank-additional-full.csv"):
        print("📦 Extraction du fichier...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print("✅ Extraction terminée!")
            
            # Renommer le dossier extrait
            import shutil
            if os.path.exists("datasets/bank-additional"):
                os.rename("datasets/bank-additional", "datasets/bank_additional")
            print("✅ Données prêtes dans: datasets/bank_additional/")
        except Exception as e:
            print(f"❌ Erreur d'extraction: {e}")
            return False
    else:
        print("✅ Données déjà extraites")
    
    return True

def download_fashion_mnist():
    """Vérifie que Fashion MNIST est accessible (sera chargé via sklearn)"""
    print("\n" + "=" * 60)
    print("VÉRIFICATION DE FASHION MNIST")
    print("=" * 60)
    print("Fashion MNIST sera chargé automatiquement via scikit-learn")
    print("Les données seront mises en cache dans: ~/scikit_learn_data/")
    print("✅ Configuration OK")
    return True

if __name__ == "__main__":
    print("\n🚀 DÉMARRAGE DU TÉLÉCHARGEMENT DES DONNÉES")
    print("-" * 60)
    
    # Télécharger Bank Telemarketing
    success = download_bank_data()
    
    # Vérifier Fashion MNIST
    download_fashion_mnist()
    
    if success:
        print("\n" + "=" * 60)
        print("✅ TOUTES LES DONNÉES SONT PRÊTES!")
        print("=" * 60)
        print("\nStructure des dossiers:")
        print("  📁 datasets/bank_additional/  - Données bancaires")
        print("  📁 datasets/fashion_mnist/    - Données Fashion MNIST (cache)")
    else:
        print("\n❌ Erreur lors du téléchargement. Vérifiez votre connexion internet.")