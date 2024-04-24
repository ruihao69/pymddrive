from enum import Enum, unique

@unique
class FloquetablePulses(Enum):
    MORLET = "Morlet"
    MORLET_REAL = "MorletReal"
    SINE_SQUARE_PULSE = "SineSquarePulse"
    COSINE = "CosinePulse"
    SINE = "SinePulse"
    EXPONENTIAL = "ExponentialPulse"