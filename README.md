# TP 3 Sémantique distributionnelle

## Consignes : Correction des biais
Énoncé volontairement très ouvert. A partir d'un corpus assez grand, il s'agit de calculer des représentations distributionnelles de type GloVe, puis de chercher des cas de biais avec la méthode analogique (roi-homme+femme=?). Dans un second temps, en modifiant le corpus de départ (par exemple en supprimant les occurrences les plus biaisées, ou en ajoutant des phrases pour contrebalancer les biais), on vérifie en construisant une nouvelle matrice GloVe que le biais est impacté par le corpus d'apprentissage.

## Execution du programme et tests

### Lancer avec uv

```
uv run main.py <key_token> -a <token_a> -b <token_b> --vectors <path_to_vectors_file>
```

### Lancer l'interpreteur ipython

```
uv run \
  --with ipython \
  ipython
```

## Méthodologie

Usage de <https://github.com/stanfordnlp/GloVe> pour calculer les plongements (recommandé par Achille Lacroix).

### Choix du corpus
<!---->
<!-- Pour obtenir un corpus de plus d'un milion de tokens : -->
<!-- Hugging Face -->
<!-- filtres : -->
<!--  * Languages = french -->
<!--  * Size = 1M - 10M -->
<!--  * Format = text -->
<!---->
<!-- <https://huggingface.co/datasets/FrancophonIA/journal_officiel> -->
<!---->
<!-- Usage dans python : -->
<!-- ```python -->
<!-- from datasets import load_dataset -->
<!---->
<!-- # Login using e.g. `huggingface-cli login` to access this dataset -->
<!-- ds = load_dataset("FrancophonIA/journal_officiel") -->
<!-- " ".join(ds["train"]["text"][:10000000]) #10 000 000 en comptant moins de 10 caractères par token -->
<!-- ``` -->
<!---->

corpus de sous-titres
<https://opus.nlpl.eu/datasets/OpenSubtitles?pair=fr&en>
corpus de sous-titres monolingues fr datant de 2016 (on peut s'attendre à des données plus biaisées en genre que sur des corpus plus récents (en particulier contenant des sous-titres de films post-2017))

nettoyage du corpus (suppression de la ponctuation puis normalisation de la casse) :

```bash
cat corpus.txt | perl -CSDA -pe 's/\p{P}+/ /g' | perl -CSDA -Mutf8 -pe '$_ = lc($_)' > corpus_net.txt
```

extraction du vocabulaire :

```bash
./GloVe/build/vocab_count -min-count 10 < corpus_net.txt > vocab.txt
```

calcul des cooccurrences :

```bash
./GloVe/build/cooccur -vocab-file vocab.txt -memory 14.0 < corpus_net.txt > cooccurrences.bin
```

melange :

```bash
./GloVe/build/shuffle < cooccurrences.bin > cooccurrences.shuf.bin
```

calcul du modele :

```bash
./GloVe/build/glove -input-file cooccurrences.shuf.bin -vocab-file vocab.txt -save-file vectors -threads 10 
```

## Cas interressants

```python
find_knn_analogy("ministre", "homme", "femme", vectors_dict)
find_knn_analogy("chef", "homme", "femme", vectors_dict)
find_knn_analogy("pute", "femme", "homme", vectors_dict)
find_knn_analogy("mère", "femme", "homme", vectors_dict)
find_knn_analogy("père", "homme", "femme", vectors_dict)
find_knn_analogy("lesbienne", "femme", "homme", vectors_dict)
find_knn_analogy("gay", "homme", "femme", vectors_dict)
```
