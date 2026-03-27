# TP 2 Sémantique distributionnelle

Code pour le TP 2

## Lancer avec uv

```
uv run main.py <chemin_corpus> -w <taille_fenetre> -k <lemme_cible>
```
## Lancer l'interpreteur ipython

```
uv run \
  --with ipython \
  --with spacy \
  --with https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.7.0/fr_core_news_sm-3.7.0-py3-none-any.whl \
  ipython
```

## À faire

Les fonctions de lissage ne sont pas encore implémentées.
