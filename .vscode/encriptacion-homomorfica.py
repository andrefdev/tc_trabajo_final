import random
from sympy import isprime, primitive_root


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

# Parámetros del criptosistema
# Generar p y g
p, g = generate_p_and_g()
# p es el "número primo grande"
# g es el "generador del grupo Z_p*"
q = (p - 1) // 2  # Subgrupo primo de Z_p*

# Generar claves
sk, pk = generate_keys(p, g, q)

# Mensaje a encriptar
mensaje = 7

# Encriptar mensaje, c1 y c2 textos cifrados que
# representan un mensaje m encriptado.
c1, c2 = encrypt(mensaje, pk, p, g, q)
print("Texto cifrado:", (c1, c2))

# Desencriptar mensaje con los textos cifrados c1 y c2
mensaje_recuperado = decrypt(c1, c2, sk, p)
print("Mensaje recuperado:", mensaje_recuperado)

# Adición homomórfica
c1_1, c2_1 = encrypt(3, pk, p, g, q)  # Encriptar 3
c1_2, c2_2 = encrypt(4, pk, p, g, q)  # Encriptar 4

c1_sum, c2_sum = homomorphic_addition(c1_1, c2_1, c1_2, c2_2, p)
# Desencriptar la suma homomórfica
resultado_suma = decrypt(c1_sum, c2_sum, sk, p)
print("Resultado de la suma homomórfica desencriptada:", resultado_suma)
"""
La suma homomórfica se comporta como una multiplicación en el espacio encriptado
es por ello que con los numeros encriptados, 3 y 4 la respuesta es 12(3*4)
"""