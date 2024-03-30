from enum import Enum, unique

@unique
class FloquetablePulses(Enum):
    MORLET = "Morlet"
    MORLET_REAL = "MorletReal"
    SINE_SQUARED_PULSE = "SineSquaredPulse"
    COSINE = "CosinePulse"
    SINE = "SinePulse"
    EXPONENTIAL = "ExponentialPulse"