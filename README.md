# coEnergy — Identification des paramètres thermiques des bâtiments

## Présentation du projet

Ce projet porte sur l'**identification de paramètres thermiques de bâtiments à partir de données simulées**. Chaque simulation fournit deux séries temporelles de 480 observations : la température intérieure $T$ et la sollicitation thermique $U$.

L'objectif est d'estimer six indicateurs : les coefficients de déperdition thermique **HLC, HLCext et HLCmit**, ainsi que les constantes de temps **τb, τn et τ∞**. Il s'agit d'un problème de **régression de paramètres physiques**, et non de prévision des prochains instants de température.

La relation physique suivante est intégrée au modèle :

$$
HLC = HLCext + HLCmit.
$$

Les réseaux estiment donc **cinq paramètres indépendants** (HLCext, HLCmit, τb, τn, τ∞). Le HLC total est ensuite reconstruit par addition, afin de respecter cette relation.

## Démarche expérimentale

J'ai organisé le travail en deux campagnes complémentaires.

**1. Expérimentations préliminaires — 500 000 simulations.** J'ai commencé par un perceptron multicouche (**MLP**) utilisant des caractéristiques statistiques. J'ai ensuite testé un **CNN 1D**, puis un **CNN-GRU**, afin d'étudier l'intérêt des informations temporelles. L'exploration a également porté sur un **réseau convolutif temporel (TCN)**, fondé sur des convolutions dilatées et des connexions résiduelles. Ces essais ont permis de comparer plusieurs représentations et de préparer les expériences finales.

**2. Expérimentations finales — 1 000 000 simulations.** J'ai repris les architectures sur des partitions communes, puis étudié une **fusion CNN-GRU + MLP** qui combine les séries temporelles et les caractéristiques statistiques. Deux pondérations de la fonction de perte ont été comparées : **50/50**, équilibrée entre les coefficients de déperdition et les constantes de temps, et **30/70**, destinée à accorder davantage d'importance aux constantes de temps.

Les six configurations finales sont :

| Modèle | Données en entrée | Principe |
|---|---|---|
| MLP | 58 caractéristiques statistiques | Modèle de référence |
| CNN 1D | Séries $T$ et $U$ | Extraction de motifs temporels locaux |
| CNN-GRU | Séries $T$ et $U$ | Convolutions et dépendances séquentielles |
| TCN | Séries et caractéristiques temporelles enrichies | Convolutions dilatées et blocs résiduels |
| Fusion 50/50 | Séries et 58 caractéristiques | Fusion CNN-GRU + MLP, perte équilibrée |
| Fusion 30/70 | Séries et 58 caractéristiques | Même fusion, perte davantage pondérée sur les constantes de temps |

Les premiers essais et les runs finaux sont présentés séparément dans le rapport : leurs jeux de test et certaines conditions expérimentales ne sont pas identiques.

## Données et protocole

Les données brutes sont stockées dans `quentin_house/`, sous la forme de **20 fichiers `.pt` de 50 000 simulations**. L'analyse exploratoire et le prétraitement sont réalisés dans les notebooks du dossier `data_analysis/`.

Le prétraitement produit deux représentations :

- les séries temporelles, de dimension `(N, 2, 480)`, utilisées par les modèles temporels ;
- les caractéristiques statistiques, initialement au nombre de 59, dont **58 sont retenues** après exclusion d'une caractéristique constante sur le train.

Les paramètres de normalisation sont calculés **uniquement sur l'ensemble d'entraînement**, puis réutilisés pour la validation et l'évaluation. Les mêmes partitions sont conservées pour tous les modèles finaux :

| Partition | Simulations | Utilisation |
|---|---:|---|
| Entraînement | 719 885 | Apprentissage des poids et normalisation |
| Validation | 89 985 | Sélection du meilleur modèle |
| Test principal | 89 987 | Comparaison des performances |
| Réserve U | 100 143 | Évaluation sur des profils de sollicitation non vus |

La réserve U comprend **287 profils de sollicitation exclus** des autres partitions. Elle permet d'étudier la généralisation à de nouvelles sollicitations simulées, sans constituer une validation sur des bâtiments réels indépendants.

## Organisation des fichiers

```text
coenergy_assessment/
├── configs/                  # Configurations des six expériences
├── data/
│   ├── prepared/             # Séries, caractéristiques, cibles et normalisations
│   └── splits/               # Partitions communes
├── data_analysis/            # Analyse exploratoire et prétraitement
├── model_experiments/        # Notebooks MLP, CNN, CNN-GRU, TCN et fusions
├── quentin_house/            # Simulations brutes (.pt)
├── src/coenergy/
│   ├── data.py               # Chargement des données et mini-batchs
│   ├── models.py             # Architectures des réseaux
│   ├── loss.py               # Fonction de perte et reconstruction physique
│   ├── training.py           # Entraînement et checkpoints
│   ├── evaluation.py         # Métriques et diagnostics
│   └── plotting.py           # Visualisation des résultats
├── results/                  # Résultats des entraînements
├── reports/                  # Rapport scientifique LaTeX et figures
├── figures/                  # Figures complémentaires du projet
├── scripts/                  # Scripts utilitaires facultatifs
├── coenergy_modules_kaggle.zip
├── pyproject.toml
└── README.md
```

### Exécution sur un ordinateur local

Le projet utilise `uv` pour gérer l'environnement Python et les dépendances.

Depuis la racine du projet, exécuter :

```bash
uv sync
```

Cette commande prépare l'environnement virtuel `.venv` et installe les dépendances définies dans `pyproject.toml`, en tenant compte du fichier `uv.lock`.

Les notebooks se trouvent dans `model_experiments/`. Ils utilisent les modules Python de `src/coenergy/` et les données préparées stockées dans `data/`.

Les étapes d'analyse exploratoire et de prétraitement sont documentées séparément dans `data_analysis/`.

### Sur Kaggle avec GPU

Le fichier **`coenergy_modules_kaggle.zip`** regroupe les modules Python de `src/coenergy/`. Son rôle est de rendre le **même code de chargement, de modélisation, d'entraînement et d'évaluation** disponible dans Kaggle, sans copier les modules un par un et sans inclure les gros tableaux de données.

Pour exécuter une expérience :

1. Importer le notebook `.ipynb` dans Kaggle.
2. Ajouter en **Input** les données préparées (`.npy`, `.json`, `.npz`) et `coenergy_modules_kaggle.zip`. Ils peuvent appartenir au même dataset Kaggle.
3. Activer le **GPU** dans les paramètres du notebook.
4. Vérifier les chemins des données et des modules dans la première cellule Python. Kaggle peut présenter le ZIP comme une archive ou comme un dossier déjà extrait.
5. Exécuter les cellules dans l'ordre, puis récupérer les résultats depuis `/kaggle/working/`.

**Important :** `coenergy_modules_kaggle.zip` contient seulement le **code Python**, et non les simulations brutes ou les tableaux `.npy`. Pour lancer un entraînement sur Kaggle, **les données préparées doivent être ajoutées séparément**.

## Résultats et évaluation

Chaque expérience sauvegarde notamment les meilleurs poids (`best.pt`), le dernier checkpoint (`last.pt`), la configuration, l'historique d'apprentissage (`history.csv`), les métriques (`metrics.csv`), les prédictions et les figures. Les résultats sont organisés par modèle et par run, par exemple :

```text
results/
├── mlp/run_001/
├── cnn/run_001/
├── cnn_gru/run_001/
├── tcn/run_001/
├── fusion_50_50/run_001/
└── fusion_30_70/run_001/
```

Les principales métriques sont **MAE**, **RMSE** et **R²**, calculées pour chacun des six indicateurs sur les partitions évaluées. Les courbes d'entraînement, les biais et la dispersion des prédictions permettent de compléter leur interprétation.

Sur la campagne finale, **la fusion 50/50** obtient les meilleurs scores $R^2$ observés sur le test principal parmi les six configurations. La pondération **30/70** n'a pas amélioré l'estimation des constantes de temps dans les essais réalisés. Ces résultats sont discutés dans `reports/rapport.tex` et `reports/rapport.pdf`, en distinguant les observations expérimentales des hypothèses explicatives.

## Rapport scientifique

Le rapport présent dans `reports/` décrit l'analyse exploratoire, les choix de prétraitement, la progression des essais préliminaires, les architectures retenues, les résultats de la campagne finale et leurs limites. Il examine notamment les difficultés d'estimation des constantes de temps et l'intérêt de la contrainte physique sur le HLC.

Le projet repose sur des **simulations** ; ses résultats ne doivent pas être interprétés comme une validation directe sur des mesures de bâtiments réels.
