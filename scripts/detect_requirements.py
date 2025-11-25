import ast
import os
import re
import chardet
import sys

def is_binary_file(file_path, sample_size=8000):
    """Détecte si un fichier est binaire en examinant un échantillon."""
    try:
        with open(file_path, 'rb') as file:
            sample = file.read(sample_size)
            # Vérifie la présence de caractères nuls ou autres indicateurs binaires
            if b'\x00' in sample:
                return True
            # Utilise chardet pour détecter l'encodage
            result = chardet.detect(sample)
            # Si la confiance est faible ou encodage inconnu, considérer comme binaire
            if result['confidence'] < 0.7:
                return True
            return False
    except Exception as e:
        print(f"Erreur lors de la vérification du fichier {file_path}: {str(e)}")
        return True  # En cas de doute, considérer comme binaire

def detect_encoding(file_path, default='utf-8'):
    """Détecte l'encodage d'un fichier."""
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)  # Lire un échantillon pour la détection
            result = chardet.detect(raw_data)
            encoding = result['encoding'] if result['confidence'] > 0.7 else default
            return encoding
    except Exception as e:
        print(f"Erreur lors de la détection de l'encodage de {file_path}: {str(e)}")
        return default

def extract_imports_with_ast(file_path):
    """Extrait les imports en utilisant l'AST avec gestion des différents encodages."""
    if is_binary_file(file_path):
        print(f"Ignoré: {file_path} (fichier binaire)")
        return set()
        
    encoding = detect_encoding(file_path)
    
    try:
        with open(file_path, 'r', encoding=encoding) as file:
            try:
                content = file.read()
                tree = ast.parse(content)
            except SyntaxError as e:
                print(f"Erreur de syntaxe dans {file_path}: {str(e)}")
                return set()
            except Exception as e:
                print(f"Erreur lors de la lecture de {file_path} avec l'encodage {encoding}: {str(e)}")
                return set()
                
        imports = set()
        
        for node in ast.walk(tree):
            # Import standard (import x, import x.y)
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.add(name.name.split('.')[0])
            # Import from (from x import y)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
                    
        return imports
    except Exception as e:
        print(f"Erreur inattendue avec {file_path}: {str(e)}")
        return set()

def scan_directory(directory='.'):
    """Scanne un répertoire pour trouver tous les imports."""
    all_imports = set()
    
    for root, dirs, files in os.walk(directory):
        # Ignorer les répertoires cachés et les environnements virtuels
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', 'env', '__pycache__', 'node_modules')]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                print(f"Analyse de {file_path}...")
                imports = extract_imports_with_ast(file_path)
                all_imports.update(imports)
                
    return all_imports

def filter_standard_library(imports):
    """Filtre les modules de la bibliothèque standard."""
    stdlib_modules = set(sys.builtin_module_names)
    
    # Modules standards additionnels fréquemment utilisés
    common_stdlib = {
        'abc', 'argparse', 'array', 'ast', 'asyncio', 'base64', 'collections', 
        'concurrent', 'contextlib', 'copy', 'csv', 'ctypes', 'datetime', 'decimal', 
        'difflib', 'distutils', 'email', 'enum', 'fileinput', 'fnmatch', 'fractions', 
        'functools', 'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http', 
        'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json', 'logging', 
        'math', 'mimetypes', 'multiprocessing', 'netrc', 'numbers', 'operator', 
        'os', 'pathlib', 'pickle', 'pkgutil', 'platform', 'pprint', 'queue', 'random', 
        're', 'shutil', 'signal', 'socket', 'sqlite3', 'ssl', 'stat', 'string', 
        'struct', 'subprocess', 'sys', 'tempfile', 'textwrap', 'threading', 'time', 
        'timeit', 'trace', 'traceback', 'types', 'typing', 'unittest', 'urllib', 
        'uuid', 'warnings', 'weakref', 'xml', 'zipfile', 'zlib'
    }
    
    stdlib_modules.update(common_stdlib)
    
    return {imp for imp in imports if imp not in stdlib_modules}

def get_installed_version(package_name):
    """Tente d'obtenir la version installée d'un package."""
    try:
        import importlib.metadata
        return importlib.metadata.version(package_name)
    except (ImportError, importlib.metadata.PackageNotFoundError):
        try:
            # Fallback pour Python < 3.8
            import pkg_resources
            return pkg_resources.get_distribution(package_name).version
        except (ImportError, pkg_resources.DistributionNotFound):
            return "Non installé ou version inconnue"

if __name__ == "__main__":
    # Vérifier si un chemin a été fourni en argument
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    else:
        project_dir = "."  # Répertoire courant par défaut
    
    print(f"Analyse des dépendances dans: {os.path.abspath(project_dir)}")
    
    # Installer chardet si nécessaire
    try:
        import chardet
    except ImportError:
        print("Installation de la bibliothèque 'chardet' pour la détection d'encodage...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
        import chardet
    
    imports = scan_directory(project_dir)
    third_party_imports = filter_standard_library(imports)
    
    print("\n=== Bibliothèques tierces utilisées ===")
    if not third_party_imports:
        print("Aucune bibliothèque tierce détectée.")
    else:
        # Trier par ordre alphabétique et tenter d'obtenir les versions
        for imp in sorted(third_party_imports):
            version = get_installed_version(imp)
            print(f"- {imp} (version: {version})")
    
    # Génération du fichier requirements.txt
    if third_party_imports:
        print("\nGénération du fichier requirements.txt...")
        with open("requirements_detected.txt", "w") as req_file:
            for imp in sorted(third_party_imports):
                version = get_installed_version(imp)
                if version != "Non installé ou version inconnue":
                    req_file.write(f"{imp}=={version}\n")
                else:
                    req_file.write(f"{imp}\n")
        print("Fichier requirements_detected.txt généré avec succès!")