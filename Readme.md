# batch_cadrage_excel.py

Un outil Python pour recadrer et redimensionner automatiquement des images de produits à partir de fichiers Excel Shopify, avec détection intelligente des objets et standardisation des marges.

## 🎯 Fonctionnalités

- **Détection automatique des objets** : Identifie et isole les produits dans les images
- **Standardisation des marges** : Applique des marges uniformes basées sur une image de référence
- **Traitement par lots** : Traite toutes les images listées dans un fichier Excel
- **Multi-onglets** : Parcourt automatiquement tous les onglets d'un classeur Excel
- **Redimensionnement intelligent** : Redimensionne proportionnellement et centre les images
- **Homogénéisation du fond** : Unifie la couleur de fond pour un rendu professionnel
- **Téléchargement automatique** : Récupère les images directement depuis les URLs

## 📋 Prérequis

```bash
pip install numpy pandas pillow requests openpyxl
```

## 🚀 Utilisation

### Syntaxe de base

```bash
python batch_cadrage_excel.py fichier.xlsx [OPTIONS]
```

### Exemples

**Traiter tous les onglets d'un classeur :**
```bash
python batch_cadrage_excel.py catalogue.xlsx
```

**Traiter un onglet spécifique :**
```bash
python batch_cadrage_excel.py catalogue.xlsx --sheet "Produits 2024"
```

**Personnaliser la taille de sortie :**
```bash
python batch_cadrage_excel.py catalogue.xlsx --target-size 800x1000
```

**Ajuster les marges :**
```bash
python batch_cadrage_excel.py catalogue.xlsx --margin-scale 0.8
```

**Désactiver le redimensionnement :**
```bash
python batch_cadrage_excel.py catalogue.xlsx --target-size none
```

## ⚙️ Options

| Option | Défaut | Description |
|--------|--------|-------------|
| `--sheet` | `all` | Nom de l'onglet à traiter ou `all` pour tous les onglets |
| `--output-dir` | `outputs_october` | Dossier de destination des images traitées |
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

1. **Lecture du fichier Excel** : Charge les onglets et extrait les URLs
2. **Téléchargement de la référence** : Récupère l'image de référence pour calculer les marges
3. **Traitement par image** :
   - Téléchargement de l'image
   - Détection automatique de l'objet
   - Recadrage avec marges standardisées
   - Homogénéisation du fond
   - Redimensionnement final
   - Sauvegarde dans le dossier de sortie

## 📁 Structure de sortie

```
outputs_october/
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
- Analyse des niveaux de gris
- Détection par différence de couleur avec le fond
- Filtrage des groupes continus de pixels
- Sélection de la bbox la plus pertinente

## ⚠️ Gestion des erreurs

- Les onglets sans colonnes requises sont ignorés avec un avertissement
- Les images en erreur sont signalées mais n'interrompent pas le traitement
- Les noms de fichiers dupliqués sont automatiquement renommés

## 💡 Conseils

- **Marges trop grandes ?** Réduisez `--margin-scale` (ex: 0.7)
- **Marges trop petites ?** Augmentez `--margin-scale` (ex: 1.1)
- **Fond non homogène ?** La couleur de fond par défaut est `(250, 248, 246)` et peut être ajustée dans le code

## 📝 Version

**v1.6** - Multi-onglets + redimensionnement + fond homogène

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