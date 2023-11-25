import pandas as pd
import numpy as np
import random
import math
from sympy import isprime, primitive_root

from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise.prediction_algorithms.knns import KNNWithZScore


# =======================ENCRIPTACION=======================

# Funcion para generar p y g
def generate_p_and_g():
    while True:
        # Generar un número primo aleatorio grande para p
        p = random.randint(10 ** 28, 10 ** 29)
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
    inv_c1 = pow(c1, -sk, p)  # Inverso multiplicativo de c1: c1^-sk mod p
    m = (c2 * inv_c1) % p  # m = c2 * c1^-sk mod p
    return m


# Función para realizar adición homomórfica entre dos textos cifrados (c1_1, c2_1) y (c1_2, c2_2)
def homomorphic_addition(c1_1, c2_1, c1_2, c2_2, p):
    c1_sum = (c1_1 * c1_2) % p  # Multiplicación de c1: c1_1 * c1_2 mod p
    c2_sum = (c2_1 * c2_2) % p  # Multiplicación de c2: c2_1 * c2_2 mod p
    return c1_sum, c2_sum


# =======================RECOMENDACIONES=======================

# DataFrame con datos de Amazon (Reemplazar datos)
data = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5],
    'product_id': [101, 102, 103, 104, 105],
    'category': ['Electronics', 'Books', 'Products', 'Cleaning', 'Clothing'],
    'price': [100, 20, 120, 25, 50],
    'rating': [5, 4, 3, 4, 5], # Los datos deben ser enteros para fin de nuestro esquema
    'reviews': [10, 5, 12, 7, 8]
})

reader = Reader(rating_scale=(1, 5))

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
    print(vector1['rating']) # 5.0
    c1_1, c1_2 = encrypt(vector1['rating'], pk, p, g, q)
    c2_1, c2_2 = encrypt(vector1['reviews'], pk, p, g, q)
    # Tupla de tuplas que almacena el vector cifrado
    vector1_encrypted = (c1_1, c1_2), (c2_1, c2_2)

    c1_1, c1_2 = encrypt(vector2['rating'], pk, p, g, q)
    c2_1, c2_2 = encrypt(vector2['reviews'], pk, p, g, q)
    vector2_encrypted = (c1_1, c1_2), (c2_1, c2_2)

    # Calcular el producto escalar homomórfico
    dot_product_1, dot_product_2 = homomorphic_addition(
        vector1_encrypted[0][0] * vector2_encrypted[0][0],
        vector1_encrypted[0][1] * vector2_encrypted[0][1],
        vector1_encrypted[1][0] * vector2_encrypted[1][0],
        vector1_encrypted[1][1] * vector2_encrypted[1][1],
        p
    )

    # Calcular las magnitudes homomórficas
    magnitude1_1, magnitude1_2 = homomorphic_addition(
        vector1_encrypted[0][0] * vector1_encrypted[0][0],
        vector1_encrypted[0][1] * vector1_encrypted[0][1],
        vector1_encrypted[1][0] * vector1_encrypted[1][0],
        vector1_encrypted[1][1] * vector1_encrypted[1][1],
        p
    )
    magnitude2_1, magnitude2_2 = homomorphic_addition(
        vector2_encrypted[0][0] * vector2_encrypted[0][0],
        vector2_encrypted[0][1] * vector2_encrypted[0][1],
        vector2_encrypted[1][0] * vector2_encrypted[1][0],
        vector2_encrypted[1][1] * vector2_encrypted[1][1],
        p
    )

    magnitude1_decrypted = decrypt(magnitude1_1, magnitude1_2, sk, p)
    magnitude2_decrypted = decrypt(magnitude2_1, magnitude2_2, sk, p)

    dot_product_decrypted = decrypt(dot_product_1, dot_product_2, sk, p)

    print(dot_product_decrypted)

    return dot_product_decrypted / (math.sqrt(magnitude1_decrypted) * math.sqrt(magnitude2_decrypted))


# Ejemplo de uso:
_vector1 = data.iloc[0]  # Obtener la primera fila como _vector1
_vector2 = data.iloc[1]  # Obtener la segunda fila como vector 2

similarity = cosine_similarity_homomorphic(_vector1, _vector2, pk, p, g, sk, q)
print("Similitud del coseno homomórfica:", similarity)

# Valores de los vectores
vector1_values = _vector1[['rating', 'reviews']].values
vector2_values = _vector2[['rating', 'reviews']].values

print("dot product: ", np.dot(vector1_values, vector2_values))
