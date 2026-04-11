Léna Baraquin

Sorbonne Nouvelle

# TP 3 — Sémantique distributionnelle

<!-- ## Consignes : correction des biais -->
<!-- Énoncé volontairement très ouvert. À partir d'un corpus assez grand, il s'agit de calculer des représentations distributionnelles de type GloVe, puis de chercher des cas de biais avec la méthode analogique (roi-homme+femme=?). Dans un second temps, en modifiant le corpus de départ (par exemple en supprimant les occurrences les plus biaisées, ou en ajoutant des phrases pour contrebalancer les biais), on vérifie, en construisant une nouvelle matrice GloVe, que le biais est impacté par le corpus d'apprentissage. -->

## Exécution du programme et tests

### Lancer avec uv

```bash
uv run main.py <key_token> -a <token_a> -b <token_b> 
  --vectors <path_to_vectors_file>
````

### Lancer l'interpréteur ipython

```bash
uv run \
  --with ipython \
  ipython
```

## Méthodologie

### Choix du corpus

J'ai choisi un corpus de sous-titres monolingues français datant de 2016 (on peut s'attendre à des données plus biaisées en genre que sur des corpus plus récents, en particulier contenant des sous-titres de films post-2017).

Le corpus est disponible à cette adresse :
[https://opus.nlpl.eu/datasets/OpenSubtitles?pair=fr&en](https://opus.nlpl.eu/datasets/OpenSubtitles?pair=fr&en)

### Traitement du corpus

Pour générer les plongements, j'ai utilisé les programmes proposés à l'adresse suivante :
[https://github.com/stanfordnlp/GloVe](https://github.com/stanfordnlp/GloVe) (recommandé par Achille Lacroix).

#### Nettoyage du corpus (suppression de la ponctuation puis normalisation de la casse)

```bash
cat corpus.txt | perl -CSDA -pe 's/\p{P}+/ /g' 
  | perl -CSDA -Mutf8 -pe '$_ = lc($_)' > corpus_net.txt
```

#### Extraction du vocabulaire

```bash
./GloVe/build/vocab_count -min-count 10 
  < corpus_net.txt > vocab.txt
```

#### Calcul des cooccurrences

```bash
./GloVe/build/cooccur -vocab-file vocab.txt -memory 14.0 
  < corpus_net.txt > cooccurrences.bin
```

#### Mélange

```bash
./GloVe/build/shuffle 
  < cooccurrences.bin > cooccurrences.shuf.bin
```

#### Calcul des plongements

```bash
./GloVe/build/glove -input-file cooccurrences.shuf.bin 
  -vocab-file vocab.txt -save-file vectors -threads 10
```

Ce programme génère un fichier `vectors.txt` contenant les tokens du vocabulaire suivis du vecteur correspondant.

### Recherche des biais

Pour chercher les potentiels biais de genre, j'ai écrit un code pour afficher les dix voisins les plus proches du vecteur correspondant à `token - homme + femme`. (cf. fichier `code.py`)

## Cas intéressants

```python
# statuts hiérarchiques

find_knn_analogy("ministre", "homme", "femme", vectors_dict)
# ['secrétaire',
#  'ministre',
#  'présidente',
#  'gouverneur',
#  'maire',
#  'directrice',
#  'procureur',
#  'doyen',
#  'nièce',
#  'cliente']

find_knn_analogy("chef", "homme", "femme", vectors_dict)
# ['secrétaire',
#  'chef',
#  'infirmière',
#  'présidente',
#  'directrice',
#  'avocate',
#  'assistante',
#  'patron',
#  'directeur',
#  'patronne']

# insultes

find_knn_analogy("pute", "femme", "homme", vectors_dict)
# ['bâtard',
#  'salaud',
#  'crétin',
#  'salopard',
#  'traître',
#  'singe',
#  'voyou',
#  'menteur',
#  'nègre',
#  'enfoiré']

# statut familial

find_knn_analogy("mère", "femme", "homme", vectors_dict)
# ['homme',
#  'père',
#  'celui',
#  'seul',
#  'enfant',
#  'garçon',
#  'fils',
#  'grand',
#  'ami',
#  'monde']

find_knn_analogy("père", "homme", "femme", vectors_dict)
# ['mère',
#  'soeur',
#  'sœur',
#  'amie',
#  'copine',
#  'femme',
#  'père',
#  'mari',
#  'fille',
#  'fiancée']

# orientation sexuelle

find_knn_analogy("gay", "homme", "femme", vectors_dict)
# ['copine',
#  'lesbienne',
#  'gay',
#  'serveuse',
#  'prostituée',
#  'nana',
#  'amie',
#  'nounou',
#  'teaseuse',
#  'fiancée']

find_knn_analogy("lesbienne", "femme", "homme", vectors_dict)
# ['hétéro',
#  'sadique',
#  'cynique',
#  'sociopathe',
#  'patriote',
#  'athlète',
#  'prédateur',
#  'suicidaire',
#  'déchet',
#  'novice']
```

## Correction des biais

Je n'ai malheureusement pas eu le temps de finir ce TP. Notamment, je ne sais pas comment supprimer les occurrences les plus biaisées du corpus avec des méthodes automatiques (le corpus étant trop grand pour supprimer ces occurrences manuellement).
