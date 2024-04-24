from enum import Enum, unique

@unique 
class ValidEnvolpeFunctions(Enum):
    UNIT = "UnitPulse"           # no envelop function (pure cosine or sine pulse)
    GAUSSIAN = "Gaussian"        # Gaussian envelop function
    SINE_SQUARE_ENVELOPE = "SineSquareEnvelope" # Sine squared envelop function