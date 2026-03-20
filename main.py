import numpy as np


def build_ppmi_matrix(corpus, window_size=2):
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


def main():
    corpus = "a b a b c".split()
    ppmi_matrix, vocab_index = build_ppmi_matrix(corpus, 2)
    __import__("pprint").pprint(ppmi_matrix)
    print(vocab_index)


if __name__ == "__main__":
    main()
