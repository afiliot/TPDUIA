# coding: latin-1

"""Implemente l'architecture ResNet18."""

from typing import Tuple

# librairie centrale: classification_models
from classification_models.tfkeras import Classifiers
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2


def ResNet18(
    classes: int,
    img_size: Tuple[int, int] = (224, 224),
    weights: str = 'imagenet',
    freeze_till: str = 'all',
    activation = 'softmax'
) -> Model:
    """
    Modèle ResNet 18.
    Paramètres:
        - classes     : nombre de classes à prédire.
        - img_size    : dimensions des images.
        - weights     : initialisation des poids: aucun (None) ou 'imagenet'.
        - freeze_till : soit None (tous les poids sont ré-entraînés), soit 'all'
                        (tous les poids sont gelés), soit le nom d'une couche en
                        particulier. Toutes les couches avant cette couche seront
                        gelées.
    """
    # import du modèle resnet grâce à la librairie resnet18.
    resnet18_, _ = Classifiers.get('resnet18')
    # définition des poids initiaux
    resnet18 = resnet18_((*img_size, 3), weights=weights)
    # spécification des poids à geler / à apprendre
    if freeze_till:
        if freeze_till == 'all':
            resnet18.trainable = False # tout est gelé!
        else:
            frozen = resnet18.get_layer(freeze_till)
            for i in range(resnet18.layers.index(frozen) + 1):
                resnet18.layers[i].trainable = False
    # on recupère le modèle jusqu'à l'antepénultième couche,
    # la sortie de la dernière couche de convolution
    embed = resnet18.layers[resnet18.layers.index(resnet18.get_layer('pool1'))] 
    resnet18_no_top = Model(
        inputs=resnet18.input,
        outputs=embed.output,
        name='ResNet18',
    ) # no_top signifie "pas de couche de classification"
    resnet18 = Sequential()
    resnet18.add(resnet18_no_top)
    # et on y ajoute une couche softmax pour la classification
    resnet18.add(Dense(
        classes,
        activation=activation,
        kernel_initializer='he_normal', name='fc1'
        )
    )
    return resnet18
