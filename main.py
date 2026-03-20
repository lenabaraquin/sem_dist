import numpy as np
import argparse
import spacy

nlp = spacy.load("fr_core_news_sm")

############################
### Chargement du corpus ###
############################


def load_corpus(corpus_path: str) -> list[str]:
    """
    Charge un corpus dans une liste de lemmes étant donné le chemin vers un fichier texte.
    """
    with open(corpus_path, "r") as f:
        corpus = f.read()
    return tokenize_corpus(corpus)


def tokenize_corpus(corpus_str: str) -> list[str]:
    """
    Retourne la liste des tokens lemmatisés du corpus.
    """
    document = nlp(corpus_str)
    corpus = [token.lemma_ for token in document]
    return corpus


#####################################
### Calcul de la matrice des ppmi ###
#####################################


def build_ppmi_matrix(corpus: list[str], window_size: int = 2):
    vocab = list(set(corpus))
    vocab_index = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)

    # matrice de cooccurrence
    M = np.zeros((V, V))

    for i, token in enumerate(corpus):
        start = max(0, i - window_size)
        end = min(len(corpus), i + window_size + 1)

        for j in range(start, end):
            if i != j:
                context = corpus[j]
                M[vocab_index[token], vocab_index[context]] += 1

    # calcul PPMI
    total = np.sum(M)
    row_sum = np.sum(M, axis=1)
    col_sum = np.sum(M, axis=0)

    PPMI = np.zeros_like(M)

    for i in range(V):
        for j in range(V):
            if M[i, j] == 0:
                continue

            p_ij = M[i, j] / total
            p_i = row_sum[i] / total
            p_j = col_sum[j] / total

            pmi = np.log2(p_ij / (p_i * p_j))
            PPMI[i, j] = max(0, pmi)

    return PPMI, vocab_index


##############################
### Lissage des embeddings ###
##############################

pass

##########################
### Similarité cosinus ###
##########################


def cosine_sim(u: np.array, v: np.array) -> np.float64:
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """

    dot_product = np.dot(u, v)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    return dot_product / (norm_u * norm_v)


def find_knn(
    key_lemma: str, ppmi_matrix: np.array, vocab_index: dict[str, int], k: int = 10
) -> list[str]:
    """
    C'est cracra mais retourne la liste des k lemmes les plus proches du lemme cible dans l'espace des plongements.
    """
    key_ppmi = ppmi_matrix[vocab_index[key_lemma]]
    sim_dict = dict()
    for lemma, index in vocab_index.items():
        similarity = cosine_sim(key_ppmi, ppmi_matrix[index])
        sim_dict[lemma] = similarity

    knn = [
        lemma
        for (lemma, _) in sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    ]
    return knn[:k]


def main():
    parser = argparse.ArgumentParser(
        description="Calcule les k plus proches voisins d'un mot cible dans un espace d'embeddings étant donné le chemin vers un corpus et une taille de fenêtre."
    )

    parser.add_argument(
        "corpus_path", help="Chemin vers le corpus (un seul fichier attendu)."
    )
    parser.add_argument(
        "-w", "--window_size", type=int, help="Taille de la fenêtre contextuelle."
    )
    parser.add_argument(
        "-k",
        "--key_lemma",
        type=str,
        help="Lemme cible, à partir duquel sont calculées les similarité cosinus avec tous les autres lemmes du corpus.",
    )

    args = parser.parse_args()

    corpus_path = args.corpus_path
    window_size = args.window_size
    key_lemma = args.key_lemma

    corpus = load_corpus(corpus_path)

    ppmi_matrix, vocab_index = build_ppmi_matrix(corpus, window_size)

    knn = find_knn(key_lemma, ppmi_matrix, vocab_index, 10)

    __import__("pprint").pprint(knn)


if __name__ == "__main__":
    main()
