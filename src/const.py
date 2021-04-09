import numpy as np

CLASSES = [
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
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
