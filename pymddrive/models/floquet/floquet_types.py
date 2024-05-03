from enum import Enum, unique

@unique
class FloquetType(Enum):
    COSINE = "Cosine"
    SINE = "Sine"
    EXPONENTIAL = "Exponential"