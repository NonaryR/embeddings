import numpy as np

def average_weights_embeddings(filters, model):
    print('Усредняем веса {} матриц весов эмбеддингов'.format(len(filters)))
    list_of_weights = []
    for l in filters:
        weights = model.get_layer('emb-{}'.format(l)).get_weights()[0]
        list_of_weights.append(weights)
    return np.mean(np.array(list_of_weights), axis=0)

def save_embeddings(embeddings, word_index, file_name):
    with open(file_name, 'w') as f:
        for word, index in word_index.items():
            f.write('\t'.join((word, str(embeddings[index]), '\n')))
    print('Готово! Загружено {} векторов слов размерностью {}.'.format(len(word_index), embeddings.shape[1]))
