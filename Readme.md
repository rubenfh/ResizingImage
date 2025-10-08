# Resizing.py

Un outil Python haute performance pour recadrer et redimensionner automatiquement des images de produits à partir de fichiers Excel Shopify, avec détection intelligente des objets, multiprocessing optimisé et standardisation des marges.

## 🎯 Fonctionnalités

- **Détection automatique des objets** : Identifie et isole les produits dans les images
- **Standardisation des marges** : Applique des marges uniformes basées sur une image de référence
- **Traitement par lots ultra-rapide** : Multiprocessing intelligent adapté à votre machine
- **Téléchargements parallèles** : Télécharge jusqu'à 20 images simultanément
- **Multi-onglets** : Parcourt automatiquement tous les onglets d'un classeur Excel
- **Redimensionnement intelligent** : Redimensionne proportionnellement et centre les images
- **Homogénéisation du fond** : Unifie la couleur de fond pour un rendu professionnel
- **Monitoring en temps réel** : Barres de progression et statistiques de performance
- **Téléchargement automatique** : Récupère les images directement depuis les URLs

## 📋 Prérequis

```bash
pip install numpy pandas pillow requests openpyxl psutil tqdm
```


## 🚀 Utilisation

### Syntaxe de base

```bash
python Resizing.py fichier.xlsx [OPTIONS]
```

### Exemples

**Traiter tous les onglets d'un classeur :**
```bash
python Resizing.py Resizing_example.xlsx
```

**Traiter un onglet spécifique :**
```bash
python Resizing.py Resizing_example.xlsx --sheet "Model 1"
```

**Personnaliser la taille de sortie :**
```bash
python Resizing.py Resizing_example.xlsx --target-size 800x1000
```

**Ajuster les marges :**
```bash
python Resizing.py Resizing_example.xlsx --margin-scale 0.8
```

**Désactiver le redimensionnement :**
```bash
python Resizing.py Resizing_example.xlsx --target-size none
```

## ⚙️ Options

| Option | Défaut | Description |
|--------|--------|-------------|
| `--sheet` | `all` | Nom de l'onglet à traiter ou `all` pour tous les onglets |
| `--output-dir` | `outputs` | Dossier de destination des images traitées |
| `--margin-scale` | `0.9` | Échelle des marges (0.0 - 1.0) |
| `--target-size` | `1520x1900` | Dimensions finales (LxH) ou `none` |

## 📊 Format du fichier Excel

Le fichier Excel doit contenir les colonnes suivantes :

- **`Reference model url`** : URL de l'image de référence (pour calculer les marges standard)
- **`Image Src`** : URLs des images à traiter

### Exemple de structure

| Reference model url | Image Src | ... |
|---------------------|-----------|-----|
| https://example.com/ref.jpg | https://example.com/img1.jpg | ... |
| | https://example.com/img2.jpg | ... |

## 🔧 Fonctionnement

1. **Configuration automatique** : Détecte les ressources système (CPU/RAM) et optimise les paramètres
2. **Lecture du fichier Excel** : Charge les onglets et extrait les URLs
3. **Téléchargement de la référence** : Récupère l'image de référence pour calculer les marges
4. **Téléchargements parallèles** : Télécharge toutes les images simultanément (10-20 threads selon votre machine)
5. **Traitement parallèle** :
   - Détection automatique de l'objet
   - Recadrage avec marges standardisées
   - Homogénéisation du fond
   - Redimensionnement final
   - Sauvegarde dans le dossier de sortie
6. **Monitoring** : Affiche les statistiques de temps et mémoire en temps réel

## 📁 Structure de sortie

```
outputs/
├── Onglet1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Onglet2/
│   ├── image1.jpg
│   └── ...
```

Chaque onglet génère un sous-dossier avec ses images traitées.

## 🎨 Algorithme de détection

Le script utilise une approche adaptative multi-critères :
- Analyse des niveaux de gris avec flou gaussien
- Détection par différence de couleur avec le fond
- Filtrage des groupes continus de pixels
- Tests multiples avec différents seuils
- Sélection de la bbox la plus pertinente

## ⚡ Performance

### Gains de performance v1.8

| Nombre d'images | Temps séquentiel | Temps parallèle | Gain |
|-----------------|------------------|-----------------|------|
| 50 images | ~8 min | ~2 min | **4x plus rapide** |
| 200 images | ~35 min | ~7 min | **5x plus rapide** |
| 500 images | ~90 min | ~18 min | **5x plus rapide** |

*Mesures sur CPU 8 cœurs, 16GB RAM*

### Configuration adaptative

Le script s'adapte automatiquement selon votre machine :

| RAM disponible | Workers CPU | Threads téléchargement |
|----------------|-------------|------------------------|
| < 8 GB | 2-4 | 10 |
| 8-32 GB | CPU-1 | 15 |
| > 32 GB | Tous les CPU | 20 |

## ⚠️ Gestion des erreurs

- Les onglets sans colonnes requises sont ignorés avec un avertissement
- Les images en erreur sont signalées mais n'interrompent pas le traitement
- Les noms de fichiers dupliqués sont automatiquement renommés
- Gestion automatique de la mémoire avec garbage collection

## 💡 Conseils

- **Marges trop grandes ?** Réduisez `--margin-scale` (ex: 0.7)
- **Marges trop petites ?** Augmentez `--margin-scale` (ex: 1.1)
- **Fond non homogène ?** La couleur de fond par défaut est `(250, 248, 246)` et peut être ajustée dans le code
- **Machine peu puissante ?** Le script s'adapte automatiquement, mais fermez les autres applications
- **Très gros volume (1000+ images) ?** Traitez par onglets séparément pour une meilleure gestion

## 📝 Version

**v1.8** - Multiprocessing optimisé + monitoring temps réel

### Nouveautés v1.8
- ✨ Multiprocessing intelligent avec configuration adaptative
- ⚡ Téléchargements parallèles optimisés
- 🎯 Traitement par lots avec chunking
- 📊 Monitoring des performances en temps réel
- 🗑️ Gestion mémoire améliorée

### Versions précédentes
- **v1.6** - Multi-onglets + redimensionnement + fond homogène

---

## 👤 Auteur

**Ruben Falvert**

## 📄 Licence

MIT License

Copyright (c) 2025 Ruben Falvert

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.