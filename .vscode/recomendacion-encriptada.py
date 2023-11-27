import pandas as pd
import numpy as np
import random
import math
import gzip
import json
from sympy import isprime, primitive_root


# =======================ENCRIPTACION=======================

# Funcion para generar p y g
def generate_p_and_g():
    while True:
        # Generar un número primo aleatorio grande para p
        p = random.randint(10 * 28, 10 * 29)
        if isprime(p):
            break

    # Encontrar un generador primitivo módulo p (g)
    """
    Ademas, permite generar un conjunto completo y único de
    claves públicas en un rango deseado [1, p - 1]
    """
    g = primitive_root(p)
    return p, g


# Función para generar claves
def generate_keys(p, g, q):
    sk = random.randint(1, q)  # Clave secreta aleatoria en Z_q*
    """
    (g^sk mod p) es preferible y más común en el criptosistema de ElGamal
    que (g^sk) debido a que el resultado se encuentra dentro del espacio 
    de residuos módulo p, asegurando que sea un valor en el rango [0, p - 1].
    (en la práctica, se utilizan representaciones modulares porquefacilitan el
     procesamiento eficiente de números grandes al limitar su tamaño a un rango específico)
    """
    pk = pow(g, sk, p)  # Clave pública: g^sk mod p,(potenciación modular.)
    return sk, pk


# Función para encriptar un mensaje m usando clave pública pk
def encrypt(m, pk, p, g, q):
    r = random.randint(1, q)  # Número aleatorio en Z_q*
    c1 = pow(g, r, p)  # c1 = g^r mod p
    c2 = (pow(pk, r, p) * m) % p  # c2 = (m * pk^r mod p) mod p
    return c1, c2


# Función para desencriptar un texto cifrado (c1, c2) usando clave secreta sk
def decrypt(c1, c2, sk, p):
    # print('c1: ', c1, 'c2: ', c2, 'sk: ', sk, 'p:', p)
    inv_c1 = pow(c1, -sk, p)  # Inverso multiplicativo de c1: c1^-sk mod p
    # print('inv_c1:', inv_c1)
    m = (c2 * inv_c1) % p  # m = c2 * c1^-sk mod p
    # print('m', m)
    return m


"""
En resumen, con ElGamal, no se puede realizar directamente una suma homomórfica como se describe.
"""

# Función para realizar adición homomórfica entre dos textos cifrados (c1_1, c2_1) y (c1_2, c2_2)
# que da como resultado al desencriptar, una multiplicacion
def homomorphic_addition(c1_1, c2_1, c1_2, c2_2, p):
    c1_sum = (c1_1 * c1_2) % p  # Multiplicación de c1: c1_1 * c1_2 mod p
    c2_sum = (c2_1 * c2_2) % p  # Multiplicación de c2: c2_1 * c2_2 mod p
    return c1_sum, c2_sum

# ======================AUXILIARES=========================

# =======================RECOMENDACIONES=======================

# Nombre del archivo
file_name = 'Magazine_Subscriptions.json.gz'

# Abrir el archivo comprimido y leer el contenido JSON
with gzip.open(file_name, 'rt', encoding='utf-8') as file:
    _data = [json.loads(line) for line in file]

df = pd.DataFrame(_data)

# Obtener solo los primeros 20 registros
data = df.head(20)

print(data.head)

"""
# DataFrame con datos de Amazon (Reemplazar datos)
data = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5],
    'product_id': [101, 102, 103, 104, 105],
    'category': ['Electronics', 'Books', 'Products', 'Cleaning', 'Clothing'],
    'price': [100, 20, 120, 25, 50],
    'rating': [5, 4, 3, 4, 5], # Los datos de calculo deben ser enteros para fin de nuestro esquema
    'reviews': [10, 5, 12, 7, 8]
})"""

# Parámetros del criptosistema
# Generar p y g
p, g = generate_p_and_g()
# p es el "número primo grande"
# g es el "generador del grupo Z_p*"
q = (p - 1) // 2  # Subgrupo primo de Z_p*

# Generar claves
sk, pk = generate_keys(p, g, q)

"""
x1, x2 = encrypt(2, pk, p, g, q)

print("####: ", decrypt(x1,x2, sk, p))
"""


# cosine_similarity_homomorphic tiene un rango de [-1,1]
def cosine_similarity_homomorphic(vector1, vector2, pk, p, g, sk, q):
    # Convertir las columnas 'rating' y 'reviews' en vectores encriptados
    # print(vector1['rating']) # 5.0

    c1_1, c1_2 = encrypt(int(pd.to_numeric(vector1['overall'])), pk, p, g, q)
    c2_1, c2_2 = encrypt(int(pd.to_numeric(vector1['vote'])), pk, p, g, q)
    # Tupla de tuplas que almacena el vector cifrado
    vector1_encrypted = (c1_1, c1_2), (c2_1, c2_2)

    """
    c1_1 = vector1_encrypted[0][0]
    c1_2 = vector1_encrypted[0][1]
    c2_1 = vector1_encrypted[1][0]
    c2_2 = vector1_encrypted[1][1]
    """

    c1_1, c1_2 = encrypt(int(pd.to_numeric(vector2['overall'])), pk, p, g, q)
    c2_1, c2_2 = encrypt(int(pd.to_numeric(vector2['vote'])), pk, p, g, q)
    # Tupla de tuplas que almacena el vector cifrado
    vector2_encrypted = (c1_1, c1_2), (c2_1, c2_2)

    """
    c1_1 = vector2_encrypted[0][0]
    c1_2 = vector2_encrypted[0][1]
    c2_1 = vector2_encrypted[1][0]
    c2_2 = vector2_encrypted[1][1]
    """

    # Calcular el producto escalar homomórfico
    # Dada una matriz A[i][j], B[i][j] dot_product_h1, dot_product_h2 representan la multiplicacion de los
    # elementos de A[i][j] y B[i][j] donde
    dot_product_h1, dot_product_h2 = homomorphic_addition(vector1_encrypted[0][0], vector1_encrypted[0][1],
                                                          vector2_encrypted[0][0], vector2_encrypted[0][1], p)
    dot_product_h3, dot_product_h4 = homomorphic_addition(vector1_encrypted[1][0], vector1_encrypted[1][1],
                                                          vector2_encrypted[1][0], vector2_encrypted[1][1], p)

    dot_product_decrypted = decrypt(dot_product_h1, dot_product_h2, sk, p) + decrypt(dot_product_h3, dot_product_h4, sk, p)
    print("dot_product decrypted: ", dot_product_decrypted)

    # Calcular las magnitudes de los vectores de la manera mas homomorfica posible
    # dadas las limitaciones operacionales del esquema ELGamal
    # Calculamos la magnitud del vector1, primero necesitaremos elevar al cuadrado los componentes
    prim_elem_cuadrado_vector1_1, prim_elem_cuadrado_vector1_2 = homomorphic_addition(vector1_encrypted[0][0],
                                                                                      vector1_encrypted[0][1],
                                                                                      vector1_encrypted[0][0],
                                                                                      vector1_encrypted[0][1], p)
    prim_elem_cuadrado_vector1_decrypted = decrypt(prim_elem_cuadrado_vector1_1, prim_elem_cuadrado_vector1_2, sk, p)
    print("prim_elem_cuadrado vector1: ", prim_elem_cuadrado_vector1_decrypted)
    seg_elem_cuadrado_vector1_1, seg_elem_cuadrado_vector1_2 = homomorphic_addition(vector1_encrypted[1][0],
                                                                                      vector1_encrypted[1][1],
                                                                                      vector1_encrypted[1][0],
                                                                                      vector1_encrypted[1][1], p)
    seg_elem_cuadrado_vector1_decrypted = decrypt(seg_elem_cuadrado_vector1_1, seg_elem_cuadrado_vector1_2, sk, p)
    print("seg_elem_cuadrado vector1: ", seg_elem_cuadrado_vector1_decrypted)
    # Calculamos la magnitud del vector2, primero necesitaremos elevar al cuadrado los componentes
    prim_elem_cuadrado_vector2_1, prim_elem_cuadrado_vector2_2 = homomorphic_addition(vector2_encrypted[0][0],
                                                                                      vector2_encrypted[0][1],
                                                                                      vector2_encrypted[0][0],
                                                                                      vector2_encrypted[0][1], p)
    prim_elem_cuadrado_vector2_decrypted = decrypt(prim_elem_cuadrado_vector2_1, prim_elem_cuadrado_vector2_2, sk, p)
    print("prim_elem_cuadrado vector2: ", prim_elem_cuadrado_vector2_decrypted)
    seg_elem_cuadrado_vector2_1, seg_elem_cuadrado_vector2_2 = homomorphic_addition(vector2_encrypted[1][0],
                                                                                      vector2_encrypted[1][1],
                                                                                      vector2_encrypted[1][0],
                                                                                      vector2_encrypted[1][1], p)
    seg_elem_cuadrado_vector2_decrypted = decrypt(seg_elem_cuadrado_vector2_1, seg_elem_cuadrado_vector2_2, sk, p)
    print("seg_elem_cuadrado vector2: ", seg_elem_cuadrado_vector2_decrypted)
    # Para despues poder sacarle sumar las componentes y obtener su raiz

    magnitud_vector1 = math.sqrt(prim_elem_cuadrado_vector1_decrypted + seg_elem_cuadrado_vector1_decrypted)
    print("magnitud_vector1 dentro fnc: ",magnitud_vector1)
    magnitud_vector2 = math.sqrt(prim_elem_cuadrado_vector2_decrypted + seg_elem_cuadrado_vector2_decrypted)
    print("magnitud_vector2 dentro fnc: ", magnitud_vector2)

    if magnitud_vector1 == 0 or magnitud_vector2 == 0:
        return 0  # Evitar división por cero
    else:
        return dot_product_decrypted / magnitud_vector1 * magnitud_vector2

# Esta funcion calcula la similaridad del coseno para todo el dataset
def calculate_homomorphic_cosine_similarity(data, pk, p, g, sk, q):
    # Obtener una lista de vectores
    vectors = [data.iloc[i] for i in range(len(data))]

    similarities = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            vector1 = vectors[i]
            vector2 = vectors[j]

            # Si la columna 'overall' del vector es NaN o nula, se coloca un 1 por default
            # para no lidiar con errores al convertir NaN a entero
            if pd.isna(vector1['overall']):
                vector1['overall'] = 1
            if pd.isna(vector1['vote']):
                vector1['vote'] = 1

            # Si la columna 'overall' del vector es NaN o nula, se coloca un 1 por default
            # para no lidiar con errores al convertir NaN a entero
            if pd.isna(vector2['overall']):
                vector2['overall'] = 1
            if pd.isna(vector2['vote']):
                vector2['vote'] = 1

            # ======================PRUEBA======================

            # función de pandas que convierte los valores de una serie a tipo numérico
            vector1_values = pd.to_numeric(vector1[['overall', 'vote']].values, errors='coerce').astype(int)
            vector2_values = pd.to_numeric(vector2[['overall', 'vote']].values, errors='coerce').astype(int)

            # Calcular la norma del vector
            print("magnitud vector1: ", np.linalg.norm(vector1_values))

            # Calcular la norma del vector

            #print("magnitud vector1: ", np.linalg.norm(vector1_values))


            # El producto punto (np.dot()) calcula el producto escalar entre dos vectores en un espacio
            # euclidiano, mientras que tu función homomorphic_addition() está realizando operaciones
            # específicas de cifrado homomórfico


            print("dot product: ", np.dot(vector1_values, vector2_values))

            # ======================PRUEBA======================

            # Calcular la similitud del coseno homomórfico entre cada par de vectores
            similarity = cosine_similarity_homomorphic(vector1, vector2, pk, p, g, sk, q)
            similarities.append(similarity)

    return similarities


# Calculamos la similitud coseno homomórfica de overall y review del dataset
similarity = calculate_homomorphic_cosine_similarity(data, pk, p, g, sk, q)
print("Similitud del coseno homomórfica:", similarity)

# En la funcion predeciremos la calificacion del usuario para un árticulo
def predict_rating(user_data, item, data, pk, p, g, sk, q):
    # Calcular la similitud del usuario con cada usuario en el conjunto de datos
    similarities = []
    for i in data.iterrows():
        # Calculamos la similitud coseno homomórfica entre el usuario dado y cada usuario en el conjunto de datos}
        # Prototipado
        similarity = calculate_homomorphic_cosine_similarity(user_data, pk, p, g, sk, q)
        similarities.append(similarity)

    # Agregar las similitudes al DataFrame
    data['similarity'] = similarities

    # Filtrar usuarios similares y que hayan calificado el ítem en cuestión
    similar_users = data[(data['similarity'] > 0) & (data['product_id'] == item['product_id'])]

    if len(similar_users) == 0:
        return 0  # Si no hay usuarios similares que hayan calificado el ítem, devolver una calificación base

    # Calcular la calificación ponderada promedio
    # Iniciamos la suma ponderada
    weighted_sum = 0
    # Iniciamos la suma de similitudes
    similarity_sum = 0
    for i, row in similar_users.iterrows():
        # Calculamos la suma ponderada de calificaciones y similitudes
        weighted_sum += row['rating'] * row['similarity']
        similarity_sum += row['similarity']

    # Calculamos la calificación ponderada promedio, evitando la división por cero
    predicted_rating = weighted_sum / similarity_sum if similarity_sum != 0 else 0
    # Devolvemos la calificación ponderada promedio como la predicción final
    return predicted_rating


# El usuario para el que quieres hacer recomendaciones
reviewerID = "A3XT9XXWXFMJ1"
reviewer_data = data[data['reviewerID'] == reviewerID]

unrated_items = data[data['reviewerID'] != reviewerID]  # Ítems no calificados por el usuario objetivo

# 3. Generar recomendaciones
recommendations = []
for i, row in unrated_items.iterrows():
    predicted_rating = predict_rating(reviewer_data, row, data, pk, p, g, sk, q)
    recommendations.append((row['asin'], predicted_rating))

# Ordenar las recomendaciones por la calificación estimada
sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

# Mostrar las mejores recomendaciones
print(f'Recomendaciones para el usuario {reviewerID}:')
for asin, rating in sorted_recommendations[:10]:
    print(f'Producto ID: {asin}, Calificación Estimada: {rating}')
