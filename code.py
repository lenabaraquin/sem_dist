import numpy as np
import argparse

###############################
### Chargement des vecteurs ###
###############################


def load_vectors(path: str, dimensions: int = 50) -> dict[str, tuple[np.ndarray]]:
    """
    Charge un dictionnaire de vecteurs étant donné le chemin vers un fichier texte.
    """
    output = dict()
    with open(path, "r") as f:
        vectors_file_content = f.readlines()
    for i, vector in enumerate(vectors_file_content):
        vector_list = vector.split()
        if len(vector_list) == dimensions + 1:
            token, embedding = (
                vector_list[0],
                np.array([float(coord) for coord in vector_list[1:]]),
            )
            output[token] = embedding

        else:
            raise ValueError(
                f"Line {i + 1}: expected {dimensions} dimensions, got {len(vector_list) - 1}"
            )

    return output


##########################
### Similarité cosinus ###
##########################


def cosine_sim(u: np.ndarray, v: np.ndarray) -> np.float64:
    """
    Calcule la similarité cosinus entre deux vecteurs.
    """

    dot_product = np.dot(u, v)

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    return dot_product / (norm_u * norm_v)


def find_knn(
    key_vector: np.ndarray,
    vectors_dict: dict[str, np.ndarray],
    k: int = 10,
) -> list[str]:
    """
    C'est cracra mais retourne la liste des k lemmes les plus proches du lemme cible dans l'espace des plongements.
    """
    sim_dict = dict()
    for token, vector in vectors_dict.items():
        similarity = cosine_sim(key_vector, vector)
        sim_dict[token] = similarity

    knn = [
        lemma
        for (lemma, _) in sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)
    ]
    return knn[:k]


def find_knn_analogy(
    key_token: str, token_1: str, token_2: str, vectors_dict: dict[str, np.ndarray]
) -> list[str]:
    if key_token not in vectors_dict:
        raise Exception("Bad token (key_token), not in vocabulary")
    if token_1 not in vectors_dict:
        raise Exception("Bad token (token_1), not in vocabulary")
    if token_2 not in vectors_dict:
        raise Exception("Bad token (token_2), not in vocabulary")
    key_vector = vectors_dict[key_token]
    vector_1 = vectors_dict[token_1]
    vector_2 = vectors_dict[token_2]
    result_vector = key_vector - vector_1 + vector_2
    return find_knn(result_vector, vectors_dict)


def _help():
    print("""
     ├╴x  vectors_dict: dict[str, ndarray]
     ├╴󰊕  load_vectors load_vectors(path: str, dimensions: int = 50) -> dict[str, tuple[np.ndarray]]: [8, 5]
     ├╴󰊕  cosine_sim cosine_sim(u: np.ndarray, v: np.ndarray) -> np.float64: [37, 5]
     ├╴󰊕  find_knn find_knn( [50, 5]
     ├╴󰊕  find_knn_analogy find_knn_analogy( [70, 5]
     ├╴󰊕  _help _help(): [86, 5]
          """)


def main():
    parser = argparse.ArgumentParser(
        description="Calcule les 10 tokens les plus proches de key_token - token_a + token b.\nPar exemple, pour `main.py roi -a homme -b femme` on attend `reine` parmi les résultats retournés."
    )

    parser.add_argument("key_token", help="Token cible.")
    parser.add_argument("-a", help="Token_a.")
    parser.add_argument("-b", help="Token_b.")
    parser.add_argument(
        "--vectors",
        required=False,
        default="vectors.txt",
        help="Chemin vers le fichier contenant les vecteurs.",
    )

    args = parser.parse_args()

    path = args.vectors
    key_token = args.key_token
    token_a = args.a
    token_b = args.b
    vectors_dict: dict = load_vectors(path)
    knn_analogy = find_knn_analogy(key_token, token_a, token_b, vectors_dict)
    print(knn_analogy)


if __name__ != "__main__":
    path = "vectors.txt"
    vectors_dict: dict = load_vectors(path)
    _help()

if __name__ == "__main__":
    main()
