# DU Intelligence Artificielle et Santé : Travaux Pratiques [Séminaire 4 - Traitement de l'image]

## Mise en place
### Synchronisation du drive `data` avec votre Drive

Avant de reprendre les procédures à réaliser pour lire chaque notebook, vous devez réaliser les choses suivantes : 

1) Ouvrir le lien de partage suivant (Google Drive): https://drive.google.com/drive/folders/12PKkp9vkuWPazhLHrwdEoBg3IDZ_T7Q_?usp=sharing
2) Le dossier contenant l'ensemble des données nécessaires aux TPs est maintenant partagé avec votre Drive. 
3) Dans le menu de gauche de Google Drive, cliquez sur "Partagés avec moi", puis doit apparaître le dossier `data`.
4) Simple clique _droit_ sur le dossier `data`, puis "Ajouter un raccourci dans Drive". 
5) Menu de gauche, aller cette fois dans "Mon Drive". Vous devriez voir le dossier `data` avec une flèche sur le pictogramme du dossier.

C'est bon pour cette étape ! :+1:

### Ouverture du notebook

Vous pouvez reprendre l'intégralité des notebooks présentés durant les sessions de travaux pratiques de ce séminaire en suivant, pour chacun des 4 TP: `image_tp1.ipynb`, `augmentation_tp2.ipynb`, `convolution_tp3.ipynb` et `covid_tp4.ipynb`, la procédure suivante : 

1) Allez sur https://colab.research.google.com/
2) Une fois sur ce site, deux possibilités : 
  - 2.1) Vous arrivez directement sur une fenêtre noir et jaune, avec inscrit sur la barre supérieure : Exemples, Récents, Google Drive, GitHub, Importer.
  - 2.2) Vous arrivez sur un notebook "Qu'est-ce que collaboratory ?".
3) Pour importer le TP qui vous intéresse, selon ces 2 possibilités, faites la chose suivante au choix : 
  - 3.1) Cliquez sur GitHub, puis comme demandé saisissez l'URL de ce GitHub: https://github.com/afiliot/TPDUIA/. Normalement, vous verrez une liste déroulante (en bas de l'écran) des 4 différents TPs cités plus haut. Sélectionnez celui qui vous intéresse. _Vous pouvez ignorer les demandes d'autorisation et autres pop-up_.
  - 3.2) Allez dans `Fichier` puis cliquez sur "importer le notebook". Vous allez alors faire appraître la fenêtre mentionnée en point 2.1. Suivez alors la procédure en 3.1.
 4) Le notebook s'affiche désormais sur votre écran. Allez alors dans `Modifier`, puis "Paramètres du notebook", et enfin sélectionnez le menu déroulant "Accélérateur matériel" puis cliquez sur "GPU".
 
C'est bon pour cette étape ! :+1:

### Point de montage avec le Drive

Dans 3 des notebooks, vous aurez à spécifier à Google Colab d'aller chercher les données sur votre drive. Pour ce faire, vous allez recontrer une cellule de code de ce type
```{python}
from google.colab import drive
drive.mount('/content/gdrive/')
```
Il faudra alors l'exécuter. Colab va vous rediriger (automatiquement ou non) vers un autre lien. Une fois sur ce nouveau site, acceptez les différentes demandes d'autorisation. Puis, copiez le code et collez-le dans le notebook. 

C'est bon pour cette étape ! :+1:

### Exécution des différentes cellules

La plupart des cellules de code s'exécutent rapidement (les données étant déjà chargées). En revanche, certains entraînements peuvent prendre de longues minutes (comme à la fin du TP `covid_tp4.ipynb`). Il se peut que l'environnement d'exécution redémarre. Dans ce cas, il faut tout reprendre depuis le début (heureusement, cela n'arrive pas souvent).

### Ressources additionnelles

Les notions abordées rapidement lors de ces TPs sont complétées par des liens (dans les cellules concernées) dirigeant vers des ressources complémentaires. D'autres ressources sont également systématiquement présentes en fin de notebook.

## Visualisation simple des notebook (sans code)

Vous pouvez également consulter les notebooks en allant sur le site : https://nbviewer.jupyter.org/
Puis, dans la barre de recherche, entrez le lien vers le notebook d'intérêt de ce GitHub, par exemple https://github.com/afiliot/TPDUIA/blob/main/TPDUIA/image_tp1.ipynb
Celui-ci devrait s'afficher de manière plus lisible. Vous pouvez aussi (lien temporaire) aller directement ici : 
- `image_tp1.ipynb` : https://nbviewer.jupyter.org/github/afiliot/TPDUIA/blob/main/TPDUIA/image_tp1.ipynb
- `augmentation_tp2.ipynb` : https://nbviewer.jupyter.org/github/afiliot/TPDUIA/blob/main/TPDUIA/augmentation_tp2.ipynb
- `convolution_tp3.ipynb` : https://nbviewer.jupyter.org/github/afiliot/TPDUIA/blob/main/TPDUIA/convolution_tp3.ipynb
- `covid_tp4.ipynb` : https://nbviewer.jupyter.org/github/afiliot/TPDUIA/blob/main/TPDUIA/covid_tp4.ipynb


## Organisation des notebook

Les notebook abordés dans le cadre de ce séminaire sont au nombre de quatre, dans l'ordre de présentation : `image_tp1.ipynb`, `augmentation_tp2.ipynb`, `convolution_tp3.ipynb` et `covid_tp4.ipynb`. Le contenu de chacun de ces notebook est détaillé ci-dessous:
- `image_tp1.ipynb` : introduction sur la notion d'image numérique - fondements physique et biologique sous-jacents, codage de l'information, représentation en Python (RVB, binaire, noir et blanc), manipulations sur les images, transformations usuelles, notion d'interpolation et de résolution.
- `augmentation_tp2.ipynb` : examples de _pipeline_ d'augmentation de données histologiques (https://doi.org/10.1038/srep27988) en TensorFlow 2.0. Test de différentes fonction d'augmentation, comparaison de l'usage de ces techniques (ou non) sur une tâche de classification de tissus.
- `convolution_tp3.ipynb` : retour sur la notion de convolution - historique de l'apprentissage profond depuis l'introduction du neurone formel, perceptron, intuitions physique et biologique de la convolution, visualisations et description de différentes opérations à l'oeuvre dans les réseaux convolutifs (pooling, convolution, couches denses, etc.). Cas d'application sur les données MNIST. 
- `covid_tp4.ipynb` : classification diagnostic de la présence d'infection pulmonaire virale SARS-CoV-2 sur données de scanner (volumiques en 3D) - chargement des données, preprocessing, augmentation, entraînement et prédiction (TensorFlow 2.0).

## Contact

N'hésitez-pas à m'envoyer un mail avec vos questions (aussi bien techniques que pratiques) à : alexandre.filiot@chru-lille.fr




