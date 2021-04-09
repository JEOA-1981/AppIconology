import numpy as np

CLASES = [
    "fondo",
    "avión",
    "bicicleta",
    "pájaro",
    "bote",
    "botella",
    "autobús",
    "carro",
    "gato",
    "silla",
    "vaca",
    "comedor",
    "perro",
    "caballo",
    "moto",
    "persona",
    "planta en maceta",
    "oveja",
    "sofá",
    "tren",
    "monitor de televisión",
]
COLORES = np.random.uniform(0, 255, size=(len(CLASES), 3))
