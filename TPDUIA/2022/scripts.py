# coding: utf-8

# librairies standard utilitaires
import os
import warnings
import itertools
from pprint import pprint
from copy import copy
from random import sample
from typing import Dict, List, Tuple, Union

# librairies de manipulation des données
import numpy as np
import pandas as pd

# librairies de traitement des images
import cv2
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
from skimage.filters import threshold_multiotsu
from skimage.transform import resize
from skimage.color import rgb2hed, hed2rgb
from skimage.util import view_as_blocks

# librairies de visualisation des images
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# openslide pour le traitement des WSI
import openslide

# outils pour faciliter la visualisation
# d'images et calculs sur notebook
from IPython.display import Image, display
from p_tqdm import p_map
from tqdm import tqdm

# deep learning
import tensorflow as tf

warnings.filterwarnings('ignore')


PATH = '/data/freganet/TPDUIA' #'/content/gdrive/MyDrive/TPDUIA'

def launch_server(wsi_path):
    print("""
        * Serving Flask app "deepzoom_server" (lazy loading)
        * Environment: production
          WARNING: This is a development server. Do not use it in a production deployment.
          Use a production WSGI server instead.
        * Debug mode: off
        * Running on http://127.0.0.1:5000/... (Press notebook interrupt to quit)
    """)
    os.system(
        f"python openslidepython/examples/deepzoom/deepzoom_server.py {wsi_path}"
    )
    print("""
        * Server stopped.
    """)
    
def print_slide_details(
        wsi: openslide.OpenSlide,
        show_thumbnail: bool = True,
        max_size: Tuple[int, int] = (1200, 800)
) -> None:
    """Informations essentielles de la lame."""
    # nombre de pixels par niveaux
    n_pixels = [w0*w1 for (w0, w1) in wsi.level_dimensions]
    n_pixels_ = tuple(format(n, ',d').replace(',',' ') for n in n_pixels)
    # taille (Gb) des niveaux
    t_pixels = tuple(round(24*n/1e9, 2) for n in n_pixels)
    # dimensions réelles (cm) de la lame
    dim_x = round(float(wsi.properties['openslide.mpp-x']) * wsi.dimensions[0] / 10000, 2)
    dim_y = round(float(wsi.properties['openslide.mpp-y']) * wsi.dimensions[1] / 10000, 2)
    # affichage des informations
    print(f"Fichier                         : {wsi}")
    print(f"Grandissement                   : {wsi.properties['openslide.objective-power']}x")
    print(f"Dimensions en pixel             : {wsi.dimensions}")
    print(f"Dimensions en cm                : {(dim_x, dim_y)}")
    print(f"Microns par pixel               : {float(wsi.properties['openslide.mpp-x']):.3f}")
    print(f"Nombre de niveaux de résolution : {wsi.level_count}")
    print(f"Grandissement des niveaux       : {wsi.level_downsamples}")
    print(f"Dimensions des niveaux          : {wsi.level_dimensions}")
    print(f"Nombre de pixels des niveaux    : {n_pixels_}")
    print(f"Taille (Gb) par niveaux         : {t_pixels}")
    print(f"Taille fichier décompressé (Gb) : {round(sum(t_pixels), 2)}")
    # optionnel: affiche une image macro de la lame
    if show_thumbnail:
        display(wsi.get_thumbnail(size=max_size))
        
from PIL.Image import Image

def get_dimensions(
    wsi: openslide.OpenSlide,
    level: int = 0,
    width: int = 512,
    height: int = 512
) -> Tuple[str, str]:
    """Calcule la dimension physique d'une région en micromètres ou centimètres."""
    physical_width = width * wsi.level_downsamples[level] * float(wsi.properties['openslide.mpp-x'])
    if physical_width / 10000 > 0.1:
        physical_width /= 10000
        physical_width = round(physical_width, 3)
        physical_width = str(physical_width)+'cm'
    else:
        physical_width = round(physical_width, 3)
        physical_width = str(physical_width)+'μm'
    physical_height = height * wsi.level_downsamples[level] * float(wsi.properties['openslide.mpp-y'])
    if physical_height / 10000 > 0.1:
        physical_height /= 10000
        physical_height = round(physical_height, 3)
        physical_height = str(physical_height)+'cm'
    else:
        physical_height = round(physical_height, 3)
        physical_height = str(physical_height)+'μm'
    return (physical_width, physical_height)

def get_region(
    wsi: openslide.OpenSlide,
    x: int, y: int, level: int, width: int, height: int
) -> Image:
    """Sélectionne la région d'intérêt sur la lame.
    Paramètres:
        x, y   : origine en haut à gauche selon la largeur et la hauteur
        level  : niveau de résolution (0 le plus élevé)
        width  : largeur de la région
        height : hauteur de la région
    """
    region = wsi.read_region((x,y), level, (width, height))
    return region

def display_region(
    wsi: openslide.OpenSlide,
    x: int, y: int, level: int, width: int, height: int
) -> None:
    """Sélectionne la région d'intérêt sur la lame et affiche les informations
    correspondantes.
    Paramètres:
        x, y   : origine en haut à gauche selon la largeur et la hauteur
        level  : niveau de résolution (0 le plus élevé)
        width  : largeur de la région
        height : hauteur de la région
    """
    region = get_region(wsi, x, y, level, width, height)
    physical_width, physical_height = get_dimensions(wsi, level, width, height)
    print(f"""
* Dimensions numériques (pixels)      : {width}p x {height}p ({format(width * height, ',d').replace(',',' ')} pixels)
* Dimensions physiques (cm ou μm)     : {physical_width} x {physical_height}
* Taille décompressée (Mo)            : {round(width*height*24/1e6, 2)}
    """)
    display(region)
    
    
def wsi_to_numpy(wsi: openslide.OpenSlide, level: int = 3) -> np.ndarray:
    """Transforme une lame format .svs en une matrice numpy selon un 
    niveau de résolution prédéfini en paramètre. Attention: le niveau
    doit être suffisamment élevé pour que l'image rentre en mémoire !
    """
    assert level > 0, 'Sélectionnez un niveau de résolution plus faible!'
    image = np.asarray(
        wsi.read_region(
            (0, 0),
            level,
            wsi.level_dimensions[level]
        ).convert('RGB')
    )
    return image


def segment_region(
    wsi: openslide.OpenSlide,
    scale_factor: float = 1/64,
    classes: int = 3,
    smooth: bool = False,
    fill_holes: bool = False,
    plot: str = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Séparer le tissu du fond blanc en utilisant la méthode de Otsu.
    Paramètres:
        scale_factor: float entre 0 et 1 quantifiant le niveau de dézoom.
                      Si égal à 1, la lame entière au plus haut niveau de 
                      résolution est chargée en mémoire !
        classes     : nombre de classes dans la méthode de seuillage d'Otsu.
                      Voir pour plus de détails:
                      https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_multiotsu.html
                      Par défaut, 3 classes de pixels sont extraits de la lames.
        smooth      : lissage du masque de segmentation (érosion puis dilatation).
                      Utile pour extraire des tissus peu denses.
        fill_holes  : remplir les "trous" du masque de segmentation.
        plot        : affichage des différentes étapes de segmentation 'all channels' ou 'lab only'
    """
    scale_factor = 1/wsi.level_downsamples[-1] if not scale_factor else scale_factor
    # choix du niveau de résolution le plus approprié en fonction du
    # zoom spécifié par l'utilisateur
    level = wsi.get_best_level_for_downsample(1/scale_factor)
    # conversion de la lame en matrice numpy
    img = wsi_to_numpy(wsi, level)
    # initialisation du masque de segmentation
    def get_mask(channel: str = 'lab') -> List[np.ndarray]:
        """Calcule un masque de segmentation.
        Si channel = 'lab', convertit au préalable l'image dans l'espace LAB.
        Sinon, utilise le canal rouge, vert ou bleu pour calculer le ou les
        seuils optimaux de séparation des différentes couches de l'image."""
        # conversion du RGB au LAB
        # plus de détails ici : https://en.wikipedia.org/wiki/CIELAB_color_space
        # et ici pour la méthode de calcul:
        # https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html#color_convert_rgb_lab
        new_img = img.copy()
        mask = np.ones((img.shape[:2]))
        # application de la méthode d'Otsu sur l'image dans l'espace de couleurs LAB
        # ou sur les canaux Rouge, Vert ou Bleu de l'espace RGB.
        if channel == 'A (LAB)':
            lab = cv2.cvtColor(new_img, cv2.COLOR_BGR2LAB)[..., 1]
            _t = threshold_multiotsu(lab, classes)[0]
        elif channel == 'rouge (RGB)':
            lab = new_img[..., 0]
            _t = threshold_multiotsu(lab, classes)[1]
        elif channel == 'vert (RGB)':
            lab = new_img[..., 1]
            _t = threshold_multiotsu(lab, classes)[1]
        elif channel == 'bleu (RGB)':
            lab = new_img[..., 2]
            _t = threshold_multiotsu(lab, classes)[1]
        # définition du masque de segmentation
        if channel == 'A (LAB)':
            mask = 1-(lab < _t)*1
        else:
            mask = (lab < _t)*1
            lab = 1 - lab
        # les pixels du fond sont codés en RGB comme (255, 255, 255)
        new_img[np.where(mask == 0)] = 255
        if smooth:
            mask = binary_closing(mask, iterations=15)
            mask = binary_opening(mask, iterations=10)
            if fill_holes:
                mask = binary_fill_holes(mask)
            new_img[np.where(mask == 0)] = 255
        return lab, mask, new_img
    # affichage des segmentations basées sur LAB, canal ROUGE, VERT puis BLEU (4x4 images)
    if plot == 'all channels':
        _, axes = plt.subplots(nrows=4, ncols=4, figsize=(27, 24))
        mag = int(wsi.properties['openslide.objective-power']) / wsi.level_downsamples[level]
        for i, channel in enumerate(['A (LAB)', 'rouge (RGB)', 'vert (RGB)', 'bleu (RGB)']):
            # on calcule le masque de segmentation sur la base de 4 canaux différents.
            lab, mask, new_img = get_mask(channel)
            axes[i, 0].imshow(img); axes[i, 0].set_axis_off()
            axes[i, 0].set_title("Niveau %d (grand. %.3f) de l'image brute" % (level, mag))
            axes[i, 1].imshow(lab); axes[i, 1].set_axis_off()
            axes[i, 1].set_title(f"Canal {channel}")
            axes[i, 2].imshow(mask, cmap='gray'); axes[i, 2].set_axis_off()
            axes[i, 2].set_title('Masque de segmentation')
            axes[i, 3].imshow(img); axes[i, 3].set_axis_off()
            axes[i, 3].imshow(mask, alpha=0.3, cmap='gray')
            axes[i, 3].set_title('Sélection des tissus')
        plt.show()
        return None
    # affichage de la segmentation basée sur LAB uniquement (1x4 images)
    else:
        lab, mask, new_img = get_mask(channel='A (LAB)')
        if plot == 'lab only':
            _, axes = plt.subplots(nrows=1, ncols=4, figsize=(27, 6))
            mag = int(wsi.properties['openslide.objective-power']) / wsi.level_downsamples[level]
            # on calcule le masque de segmentation sur la base du canal A de LAB.
            lab, mask, new_img = get_mask(channel='A (LAB)')
            axes[0].imshow(img); axes[0].set_axis_off()
            axes[0].set_title("Niveau %d (grand. %.3f) de l'image brute" % (level, mag))
            axes[1].imshow(lab); axes[1].set_axis_off()
            axes[1].set_title(f"Canal A (LAB)")
            axes[2].imshow(mask, cmap='gray'); axes[2].set_axis_off()
            axes[2].set_title('Masque de segmentation')
            axes[3].imshow(img); axes[3].set_axis_off()
            axes[3].imshow(mask, alpha=0.3, cmap='gray')
            axes[3].set_title('Sélection des tissus')
            plt.show()
        return img, lab, new_img, mask
    
from openslide.deepzoom import DeepZoomGenerator as DZG

def create_dzg(
    wsi: openslide.OpenSlide,
    wsi_path: str,
    tile_size: int,
    target_shape: int,
    overlap: int,
    thresh: float,
    scale_factor: float = 1/32
) -> DZG:
    """Créée un objet DZG avec des attributs supplémentaires utiles
    pour le traitement de la lame et des patches.
    
    Paramètres:
        wsi          : la lame en format openslide.OpenSlide
        wsi_path     : le chemin vers cette lame
        tile_size    : la taille en micromètres des patches
        target_shape : la taille en pixels des patches sauvegardés
        overlap      : le pourcentage de superposition entre les patches
        thresh       : le pourcentage maximal de fond blanc que peut contenir
                       un patch
        scale_factor : pourcentage de dezoom (défaut=1/32)
                       
    Return:
        dz : l'objet DeepZoomGenerator avec des attributs supplémentaires
    """
    # tile_size est la taille des patchs en micromètres
    # tile_size_ est la taille des patchs en pixels
    ppx = float(wsi.properties['openslide.mpp-x']) # micromètres par pixels
    tile_size_ = round(tile_size / ppx)
    # overlap est en micromètres, overlap_ en pixels
    overlap_ = round(tile_size_ * overlap)
    # on créée l'objet DZG
    dz = DZG(
        osr=wsi,
        tile_size=tile_size_,
        overlap=overlap_
    )
    # on exécute la segmentation pour récupérer le masque
    dz.img_raw, dz.lab, dz.img_new, dz.mask = segment_region(wsi, scale_factor=scale_factor, classes=3)
    # on enregistre la lame
    dz.wsi = wsi
    dz.wsi_path = wsi_path
    # tile_size est la taille des patchs en micromètres
    dz.tile_size = tile_size
    # tile_size est la taille des patchs en micromètres
    # tile_size_ est la taille des patchs en pixels
    dz.tile_size = tile_size
    dz.tile_size_ = tile_size_
    # target_shape est la taille des patchs en pixels
    # et sera la taille finale des patches une fois
    # sauvegardés
    dz.target_shape = target_shape
    # overlap_ est la taille de l'overlap en pixels
    dz.overlap = overlap
    dz.overlap_ = overlap_
    # on ajoute le seuil maximal de fond blanc que peut
    # contenir un patch
    dz.thresh = thresh
    # w_0 et h_0 sont les dimensions (largeur, hauteur)
    # de la lame au niveau de résolution le plus élevé
    dz.w_0, dz.h_0 = dz.level_dimensions[-1]
    # w_m et h_m sont les dimensions (largeur, hauteur)
    # du masque de segmentation au niveau de résolution
    # plus faible
    dz.w_m, dz.h_m = dz.mask.shape[::-1]
    # f_w et f_h sont les taux exacts de dezoom et sont 
    # calculés comme les ratios de w_0 sur w_m et 
    # h_0 sur h_m respectivement
    dz.f_w, dz.f_h = dz.w_0 / dz.w_m, dz.h_0 / dz.h_m
    # ce paramètre spécifie l'indice du plus haut niveau
    # de résolution de l'objet DZG
    dz.l0_z = dz.level_count - 1
    # ainsi que l'identifiant du patient
    dz.patient_id = os.path.split(wsi_path)[0][-4:]
    # on spécifie le dossier où seront stockées les patches
    dz.output_folder = f'{PATH}/PATCHES/{dz.patient_id}_taille-{tile_size}/'
    # on créée ce dossier s'il n'existait pas encore
    os.makedirs(dz.output_folder, exist_ok=True)
    # on ajoute le niveau de dézoom par niveau de résolution
    # de l'objet DZG
    dz.level_downsamples = [
        round(
            dz.level_dimensions[-1][0] / dzl[0], 0
        ) for dzl in dz.level_dimensions
    ]
    # on ajoute le nombre de patches par niveau de résolution
    # de l'objet DZG
    dz.level_tile_counts = [
        dzl[0] * dzl[1] for dzl in dz.level_tiles
    ]
    return dz

def print_dz_properties(dz: DZG):
    print(f'Nombre de niveaux de résolution\n{dz.level_count}\n')
    print(f'Nombre total de patchs\n{dz.tile_count}\n')
    print(f'Nombre de patchs par niveau de résolution\n{dz.level_tile_counts[::-1]}\n')
    print(f'Nombre de patchs par niveau de résolution selon (largeur, longueur)\n{dz.level_tiles[::-1]}\n')
    print(f'Dimension des niveaux de résolution\n{dz.level_dimensions[::-1]}\n')
    print(f'Niveau de dézoom par niveau de résolution\n{dz.level_downsamples[::-1]}\n')

def get_grid(dz: DZG) -> List[Tuple[int, int]]:
    """Retourne une liste de coordonnées (sur la largeur et la longueur)
    de toutes les tiles du plus haut niveau de résolution de la lame.
    Par exemple, une valeur possible de list(grid) est:
    [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]"""
    dz.w_tiles, dz.h_tiles = dz.level_tiles[dz.l0_z]
    grid = list(itertools.product(range(dz.h_tiles), range(dz.w_tiles)))
    assert len(grid) == dz.h_tiles * dz.w_tiles
    return grid

def get_regions(dz: DZG, i: int, j: int) -> Tuple[List[int], List[int]]:
    """Retourne une région d'intérêt étant donné deux indices i et j.
    Paramètres:
        i   : indice selon la largeur de la lame
        j   : indice selon la hauteur de la lame

    Outputs:
        region_0 :  liste de 4 entiers (w_0, h_0, t_w, t_h) avec w_0 et h_0
                    les coordonnées du coin en haut à gauche du patch d'indice
                    (i, j). t_w et t_h sont la largeur et la hauteur du patch
                    en pixel. En principe, t_w = t_h sauf pour les patches 
                    présents sur les bordures de la lame. Les coordonnées w_0,
                    h_0, t_w, t_h sont prises pour le plus haut niveau de 
                    résolution.
        
        region_l :  liste de 4 entiers (h_start, h_end, w_start, w_end) 
                    correpondant aux coordonnées du patch équivalent à
                    un niveau de résolution plus faible (ex. 32). Ces
                    coordonnées permettront de calculer le pourcentage 
                    de fond blanc sur le patch grâce au masque de segmen-
                    tation.
    """
    # coordonnées dans le niveau de résolution le plus élevé
    (w_0_ij, h_0_ij), _, (d_w_0_ij, d_h_0_ij) = dz.get_tile_coordinates(dz.l0_z, (j, i))
    # coordonnées dans le niveau de résolution le plus faible
    d_h_l_ij, d_w_l_ij = round(d_h_0_ij / dz.f_h), round(d_w_0_ij / dz.f_w)
    start_w_l_ij, start_h_l_ij = round(w_0_ij / dz.f_w), round(h_0_ij / dz.f_h)
    region_0 = [w_0_ij, h_0_ij, d_h_0_ij, d_w_0_ij]
    region_l = [start_h_l_ij, start_h_l_ij + d_h_l_ij, start_w_l_ij, start_w_l_ij + d_w_l_ij]
    return region_0, region_l

def plot_tiles_on_row(
        dz: DZG,
        grid: List[Tuple[int, int]],
        j: int = 200,
        n_patches: int = None
    ) -> None:
    # on récupère toutes les coordonnées de la ligne j
    row = [g for g in grid if g[0] == j]
    n_patches = len(row) if n_patches is None else n_patches
    tiles = []
    # on copie l'objet dz et on déclare un seuil de 1.01,
    # ce qui a pour effet de sélectionner toutes les patchs
    # de la ligne j
    dz_ = copy(dz);dz_.thresh = 1.01
    for coords in tqdm(row, desc=f'Extracting tiles on row {j}...'):
        out = ij_tiling(dz_, coords, plot=False)
        if out is not None:
            tile = out[0]
            tiles.append(tile)
    # on affiche les 100 premiers patches de la liste
    n = len(tiles)
    fig, axes = plt.subplots(n_patches//10, 10, figsize=(14, (n_patches//10)*(16/10)))
    for i, tile in enumerate(tiles[:n_patches]):
        ax = axes[i//10, i%10]
        ax.imshow(tile)
        ax.set_title(f'({j}, {i})')
        ax.axis('off')

def visualize_band(
    wsi: openslide.OpenSlide,
    dz: DZG,
    row: int = 200, delta: int = 200, band_level: int = 1
) -> None:
    """Paramètres
        row        : ligne d'intérêt à visualiser
        delta      : delta sert à ajuster la zone à visualiser (arbitraire)
        band_level : niveau de résolution de la bande à visualiser (défaut x20)"""

    wsi_width, wsi_height = wsi.dimensions # largeur et longueur de la lame
    tiles_on_width, tiles_on_height = dz.level_tiles[-1] # n patchs sur largeur et longueur
    # position sur la largeur du coin en haut à gauche
    band_w0 = 0
    # position sur la hauteur du coin en haut à gauche
    band_h0 = int(row/tiles_on_height * wsi_height) - delta
    # largeur de la bande
    band_width = 256 * 10
    # hauteur
    band_height = 256
    display_region(wsi, band_w0, band_h0, band_level, band_width, band_height)
    
from PIL.Image import BILINEAR

def ij_tiling(
        dz: DZG,
        coords: Tuple[int],
        plot: bool = False,
        save: bool = False
) -> Union[None, Tuple[np.ndarray, str]]:
    """Lit et stocke les patches de la lame selon des
    coordonnées (i, j) spécifiées en paramètres `coords` si
    le patch en question dispose d'un niveau de fond blanc inférieur
    à celui précisé précédemment (`dz.thresh`).
    """
    i, j = coords
    region_0, region_l = get_regions(dz, i, j)
    # lecture du masque binaire background / foreground au niveau
    # de résolution plus faible. On réutilise les coordonnées 
    # calculées avec la fonction "get_regions" !
    mask_l_ij = dz.mask[region_l[0]:region_l[1], region_l[2]:region_l[3]]
    # pourcentage de background (fond blanc)
    bgp = 1 - mask_l_ij.mean()
    # si le patch contient suffisamment de tissu, on lit puis stocke
    # le patch correspondant au niveau de résolution le plus élevé
    if bgp < dz.thresh:
        # lecture de la région d'intérêt et conversion en RGB
        tile_0_ij_ = dz.get_tile(dz.l0_z, (j, i)).convert('RGB')
        size_0_ij = tile_0_ij_.size
        if size_0_ij[0] == size_0_ij[1]:
            # re-dimensionnement selon une taille cible "target_shape"
            tile_0_ij = tile_0_ij_.resize(
                (dz.target_shape, dz.target_shape), BILINEAR
            )
        else:
            tile_0_ij = copy(tile_0_ij_)
        # nom de fichier
        tile_path = 'tilesize%d_i%d_j%d_w%d_h%d_dw%d_dh%d_overlap%f_pbg%f' % (
                tuple([dz.tile_size, i, j] + region_0 + [dz.overlap, bgp])
            )
        if save:
            # stockage du patch avec les informations utiles dans le
            # nom de fichier
            tile_path = os.path.join(dz.output_folder, tile_path + '.png')
            tile_0_ij.save(tile_path)
        # rendu graphique si plot = True
        if plot:
            # conversion de l'image en matrice numpy
            tile_0_ij = np.array(tile_0_ij)
            tile_0_ij_ = np.array(tile_0_ij_)
            # on créée la figure et les axes
            fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(17, 3))
            
            # 0: patch au niveau de résolution le plus élevé avant redimensionnement
            # en target_shape x target_shape pixels
            axes[0].imshow(tile_0_ij_)
            axes[0].set_title(
                f'% fond blanc {round(bgp, 2)}\n(dezoom {round(dz.f_h, 0)})\n({dz.tile_size}μm x {dz.tile_size}μm)\n({dz.tile_size_}p x {dz.tile_size_})'
            )
            
            # 1: patch au niveau de résolution le plus élevé après redimensionnement
            axes[1].imshow(tile_0_ij)
            axes[1].set_title(
                f'% fond blanc {round(bgp, 2)}\n(dezoom {round(dz.f_h, 0)})\n({dz.tile_size}μm x {dz.tile_size}μm)\n({dz.target_shape}p x {dz.target_shape}p)'
            )

            # 2: masque de segmentation pour le patch considéré
            axes[2].imshow(mask_l_ij, cmap='gray')
            axes[2].set_title(
                f'% fond blanc {round(bgp, 2)}\n(dezoom {round(dz.f_h, 0)})\n({mask_l_ij.shape[0]}p x {mask_l_ij.shape[1]}p)'
            )
            
            # 3: redimensionnement du masque et superposition au patch
            # affichage de la classe "fond blanc"
            mask_l_ij_resized = resize(mask_l_ij, (tile_0_ij.shape[:-1]))
            axes[3].imshow(tile_0_ij)
            axes[3].imshow(-mask_l_ij_resized, alpha=0.5, cmap='gray')
            axes[3].set_title(f'Background\n(dezoom 1.0)\n({tile_0_ij.shape[0]}p x {tile_0_ij.shape[1]}p)')
            
            # 4: idem
            # # affichage de la classe "Tissus"
            axes[4].imshow(tile_0_ij)
            axes[4].imshow(mask_l_ij_resized, alpha=0.5, cmap='gray')
            axes[4].set_title(f'Tissus\n(dezoom 1.0)\n({tile_0_ij.shape[0]}p x {tile_0_ij.shape[1]}p)')
            plt.show()
        return tile_0_ij, tile_path
    return None, None
    
    
from PIL import Image

def plot_patch_mask(dz: DZG, figsize: Tuple[int, int] = (20, 10)) -> None:
    """Affichage de la mosaïque de patches sur la lame."""
    # on récupère l'ensemble des données d'intérêt pour
    # l'affichage graphique: l'objet openslide, l'image
    # de la lame (qui rentre en mémoire!), et le masque
    # de segmentation
    wsi, image, mask = dz.wsi, dz.img_raw, dz.mask
    # on récupère également les chemins des patches
    # qui ont été créés précédemment
    tiles_names = os.listdir(dz.output_folder)
    # on calcule les dimensions de l'image (ce sont
    # les mêmes que celles du masque)
    thumb_h, thumb_w = image.shape[:2]
    # on calcule le facteur de dezoom pour l'affichage
    # de l'image finale
    dwn_w = wsi.level_dimensions[0][0] / thumb_w
    dwn_h = wsi.level_dimensions[0][1] / thumb_h
    print(dwn_h, dwn_h)
    # on initialise l'image où se superposeront les patches,
    # avec les mêmes dimensions que l'image d'origine
    img = Image.new('1', (thumb_h, thumb_w), 0)
    # on créée une figure vierge
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax = axes[1]
    # on itère sur tous les patchs et on créée à chaque fois un rectangle
    # qui délimite le patch en question (s'il a été sélectionné par l'algorithme!,
    # c'est-à-dire si sa proportion de tissus est suffisant)
    for idx, tile in tqdm(enumerate(tiles_names), total=len(tiles_names)):
        # on récupère les coordonnées du patch (niveau de résolution le 
        # plus élevé),
        # que l'on divisent ensuite par le niveau de dezoom pour convertir
        # ces coordonnées dans le niveau de résolution plus faible 
        t_s = tile.split('_')
        h_t = int(t_s[-5][1:]) / dwn_h
        w_t = int(t_s[-6][1:]) / dwn_w
        dh_t = int(t_s[-3][2:]) / dwn_h
        dw_t = int(t_s[-4][2:]) / dwn_w
        # on génère un dessin de rectangle avec les coordonnées re-dimensionnées
        # que l'on ajoute à l'image vierge `img`
        patch = patches.Rectangle((w_t, h_t), dh_t, dw_t, alpha=0.5, ec='blue')
        ax.add_patch(patch)
        ImageDraw.Draw(img).rectangle(
            [w_t, h_t, dh_t + w_t, dw_t + h_t],
            outline=1, fill=1
        )
    # on convertit l'image avec les rectangles en une matrice numpy
    # faite de 0 et 1
    tiles_mask = np.array(img) * 1.0
    # on l'affiche avec l'image brute chargée au début de la fonction
    ax.imshow(image, alpha=0.7, origin='upper');ax.axis('off')
    ax.set_title('Lame avec les patchs sélectionnés')
    ax = axes[0]
    ax.imshow(image, origin='upper');ax.axis('off')
    ax.set_title('Lame brute')
    
from PIL import Image

def render_hed(patch_path, figsize: Tuple[int] = (11, 3)) -> None:
    # lecture d'un patch
    patch = Image.open(patch_path)
    # conversion du RGB au HED
    ihc_hed = rgb2hed(patch)
    # filtrage des pixels négatifs
    ihc_hed[ihc_hed<0] = 0  
    # construction des 3 canaux H, E et D
    null = np.zeros_like(ihc_hed[:,:,0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:,:,0], null, null), axis=-1)) #Hematoxylin channel
    ihc_e = hed2rgb(np.stack((null, ihc_hed[:,:,1], null), axis=-1)) #Eosin
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:,:,2]), axis=-1)) #DAB
    # rendu graphique
    fig, axes = plt.subplots(1, 4, figsize=figsize, sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(patch)
    ax[0].set_title("Patch d'origine")

    ax[1].imshow(ihc_h)
    ax[1].set_title("Hématoxyline")

    ax[2].imshow(ihc_e)
    ax[2].set_title("Eosine")

    ax[3].imshow(ihc_d)
    ax[3].set_title("Diaminobenzidine")

    for a in ax.ravel():
        a.axis('off')
    fig.tight_layout()
    plt.show()

    
def store_data(ds: tf.data.Dataset, add_label: bool = True) -> List[np.ndarray]:
    """Visualisation des images à partir d'un objet tf.data.Dataset."""
    filenames, images, labels = [], [], []
    # on itère sur le dataset avec une boucle for
    for element in tqdm(ds):
        filename, image = element['filename'].numpy()[0].decode(), element['image'].numpy()[0]
        if add_label:
            label = element['label'].numpy()[0]
        else:
            label = None
        filenames.append(filename)
        images.append(image)
        labels.append(label)
    return np.array(filenames), np.array(images), np.array(labels)

def get_labels_distribution(labels: np.ndarray) -> pd.DataFrame:
    """Calcul des effectifs et proportions dans chaque classe."""
    distributions = pd.DataFrame(
        np.unique(labels, return_counts=True),
        index = {'Class index', 'N'}
    ).T
    distributions['p%'] = (
        distributions['N'] / distributions['N'].sum()
    ).round(3) * 100
    distributions['p%'] = distributions['p%'].astype(str).apply(lambda x: x[:4]+'%')
    distributions['Class index'] = distributions['Class index'].astype(int)
    distributions.loc['overall', :] = ['-', distributions['N'].sum(), '100.0%']
    distributions.columns = ['Class index', 'N', 'p%']
    distributions['N'] = distributions['N'].astype(int)
    return distributions

def get_train_val_test_labels_distributions(labels: List[np.ndarray]) -> pd.DataFrame:
    """Mêmes calculs que la fonction précédente mais sur les 3 jeux de labels."""
    labels_train, labels_val, labels_test = labels
    # on assemble les 3 tableaux
    distributions = get_labels_distribution(labels_train).merge(
        get_labels_distribution(labels_val), on='Class index', suffixes=['_train', '_val']
    ).merge(
        get_labels_distribution(labels_test), on='Class index', suffixes=['_test', '_test']
    )
    distributions.columns = list(distributions.columns[:-2]) + list(['N_test', 'p%_test'])
    distributions.index = [
        'tumour epithelium', 'simple stroma',
        'complex stroma', 'immune cell conglomerate',
        'debris and mucus', 'mucosal glands',
        'adipose tissue', 'background', 'overall'
    ]
    return distributions

classes_dict = {
    0: 'tumour epithelium', 1: 'simple stroma',
    2: 'complex stroma', 3: 'immune cell conglomerate',
    4: 'debris and mucus', 5: 'mucosal glands',
    6: 'adipose tissue', 7: 'background'
}

def plot_images(images: np.ndarray, labels: np.ndarray) -> None:
    """Affiche 12 images par classes de tissus."""
    fig, axes = plt.subplots(8, 12, figsize=(17, 12))
    for label in range(9):
        images_ = images[
                np.where(labels==label)[0]
        ][:12]
        for i, img in enumerate(images_):
            ax = axes[label, i]
            ax.imshow(img);ax.axis('off')
            if i == 0:
                ax.set_title(f'Classe {label}: {classes_dict[label]}')
    plt.subplots_adjust(wspace=-0.5, hspace=0.35)
    plt.show()
    
def plot_large_image(image: np.ndarray, filename: str) -> None:
    """Affiche toutes les images larges (n=10)."""
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image)
    ax.set_title(f'Image {filename}')
    plt.show()

def show_augment(
    images: np.ndarray,
    labels: np.ndarray,
    pipeline: tf.keras.Sequential = None
) -> None:
    """Affiche quelques images avant et après augmentation
    selon une pipeline d'entrée (définie par certaines opérations)/"""
    if pipeline is None:
        pipeline = geom_augmentation_pipeline
    fig, axes = plt.subplots(2, 12, figsize=(14, 4))
    for i in range(12):
        # avant augmentation
        ax = axes[0, i]
        im = images[i]
        ax.imshow(im)
        ax.axis('off')
        ax.set_title(f'Classe {labels[i]}')
        # après augmentation
        ax = axes[1, i]
        im = np.expand_dims(images[i], 0)
        ax.imshow(pipeline(im)[0, ...])
        ax.axis('off')
        ax.set_title(f'Classe {labels[i]}')
    fig.show()
    
def plot_history(history: pd.DataFrame) -> None:
    """Affiche l'évolution de l'accuracy (taux de précision)
    et de la fonction de coût (ou loss) en fonction du nombre
    d'epochs et selon l'échantillon de données."""
    # on initialise la figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    # on affiche d'abord le taux de réussite
    ax = history[['Training accuracy', 'Validation accuracy']].plot(
        ax=axes[0],
        color=['blue', 'red'],
        style=['--', '-o'],
    )
    ax.grid('on')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy on training and validation data sets')
    # puis la fonction de coût
    ax = history[['Training loss', 'Validation loss']].plot(
        ax=axes[1],
        color=['blue', 'red'],
        style=['--', '-o'],
    )
    ax.grid('on')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (Log scale)')
    ax.set_yscale('log')
    ax.set_title('Loss on training and validation data sets')
    plt.show()
    
def get_predictions_table(
    model: tf.keras.models.Model,
    images: np.ndarray,
    labels: np.ndarray,
    filenames: np.ndarray,
    add_img_coords: bool = True
) -> pd.DataFrame:
    """Retourne un data frame pandas avec, pour chaque image: les probabilités associées
    aux classes prédites, la classe prédite (maximum des probabilités) et le chemin vers
    l'image."""
    predicted_probas = model.predict_proba(images, verbose=1)
    predicted_classes = model.predict_classes(images, verbose=1)
    # on construit une table avec les probabilités et les classes prédites
    # par l'algorithme sur l'échantillon de test
    predictions = pd.DataFrame(predicted_probas, predicted_classes)
    predictions.columns = [f'probability class {i}' for i in range(8)]
    predictions['predicted class'] = predictions.index
    predictions = predictions.reset_index(drop=True)
    predictions['true class'] = labels
    predictions['filename'] = filenames
    if add_img_coords:
        predictions['large_filename'] = predictions['filename'].apply(
            lambda x: x.split('.tif')[0].split('_')[1]+'.tif'
        )
        predictions['col'] = predictions['filename'].apply(
            lambda x: x.split('_Col_')[1].split('.tif')[0]
        ).astype(int).values

        predictions['row'] = predictions['filename'].apply(
            lambda x: x.split('.tif_Row_')[1].split('_')[0]
        ).astype(int).values
    return predictions

def get_accuracies(matrix: pd.DataFrame) -> None:
    """Affiche les taux de réussite par classes."""
    for true_class in range(0, 8):
        col = matrix.iloc[:, true_class].values
        true  = col[true_class]
        false = col[[t for t in range(0, 8) if t != true_class]].sum()
        print(f'Class {classes_dict[true_class]}: {(100*true/(true+false)):.1f}%')
        
from PIL import Image

def patchify(images_large: Dict[str, np.ndarray], filenames_large: np.ndarray) -> None:
    """Partitionne chacune des images grand format de taille (5000, 5000)
    en 33 x 33 patches de taille (150, 150)."""
    for filename_large in tqdm(filenames_large):
        # 150 est un multiple de 4950 contrairement à 5000
        # si l'on garde l'image en dimension (5000, 5000),
        # view_as_blocks de skimage.util ne fonctionne pas
        image = images_large[filename_large][:4950, :4950]
        # on segmente en patch de taille (150, 150)
        patches = view_as_blocks(
            image,
            block_shape=(150, 150, 3)
        )[:, :, 0, :, :, :]
        # on rescale l'image en divisant par 255.
        patches = patches.reshape(33*33, 150, 150, 3) * (1./255)
        # on standardise les patches (centrage-réduction)
        patches = tf.image.per_image_standardization(patches).numpy()
        # on créée le dossier correspondant (au cas où)
        os.makedirs(f'{PATH}/colorectal_histology_large/2.0.0b/{filename_large}/', exist_ok=True)
        _ = np.save(f'{PATH}/colorectal_histology_large/2.0.0b/{filename_large}/patches.npy', patches)

def get_patches(filenames_large: np.ndarray) -> Dict[str, np.ndarray]:
    """Retourne un dictionnaire {nom de l'image: patches}."""
    patches = {}
    for filename_large in tqdm(filenames_large):
        patches[filename_large] = np.load(
            f'{PATH}/colorectal_histology_large/2.0.0b/{filename_large}/patches.npy'
        )
    return patches

def create_segmentation_mask(
    model: tf.keras.models.Model,
    image_large: np.ndarray,
    patches: np.ndarray
) -> List[np.ndarray]:
    """Créée un masque de segmentation à partir des prédictions
    du modèle évaluées sur les patches d'entrée d'une image grand
    format. Créée également l'image sous-dimensionnée en (1000, 1000)
    image_r et le masque également sous-dimensionné mask_r."""
    image = image_large[:4950, :4950]
    predictions = model.predict_classes(patches, verbose=1).reshape((33, 33))
    mask = np.zeros((4950, 4950)) * np.nan
    for i in range(33):
        for j in range(33):
            mask[(i*150):((i+1)*150), (j*150):((j+1)*150)] = predictions[i, j]
    mask_r = resize(mask, (1000, 1000))
    image_r = np.array(
        Image.fromarray(image).resize((1000, 1000))
    )
    return mask, mask_r, image_r

def plot_heatmap(
    image_reduced: np.ndarray,
    mask_reduced: np.ndarray,
    figsize: Tuple[int, int] = (20, 10)
) -> None:
    """Affiche l'image brute avec la carte de segmentation."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # affichage de l'image brute
    ax = axes[0]
    ax.imshow(image_reduced);ax.axis('off')
    ax.set_title("Image d'origine")
    # affichage du masque de segmentation à 8 classes
    # d'abord on affiche l'image
    ax = axes[1]
    ax.imshow(image_reduced);ax.axis('off')
    # puis le masque
    # on définit pour cela une échelle de couleurs à 8 couleurs
    cmap = matplotlib.cm.get_cmap('Accent', 8)
    # on superpose le masque
    ax.imshow(mask_reduced, cmap=cmap, alpha=0.8)
    ax.set_title(f'Distribution spatiale des classes de tissus prédites')
    # et on définit l'échelle de couleur qui viendra
    # s'afficher à droite du graphique
    # cette étape est technique puisqu'elle nécessite
    # de configurer correctement l'objet "colorbar"
    norm = matplotlib.colors.Normalize(vmin=0, vmax=8)
    cbar_ax = fig.add_axes([1.00, 0.155, 0.02, 0.8])
    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        cax=cbar_ax,
        label='Classe',
        aspect=20
    )
    cbar.set_ticks(np.linspace(0, 7, 8)+0.5)
    tick_labels = list(classes_dict.values())
    tick_texts = cbar.ax.set_yticklabels(tick_labels)
    tick_texts[0].set_verticalalignment('top')
    tick_texts[-1].set_verticalalignment('bottom')
    # on affiche
    plt.tight_layout()
    plt.show()