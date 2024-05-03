from enum import Enum, unique

@unique
class TullyOnePulseTypes(Enum):
    NO_PULSE = "NoPulse"
    ZEROPULSE = "ZeroPulse" # for debugging
    UNITPULSE = "UnitPulse" # for debugging
    PULSE_TYPE1 = "PulseType1"
    PULSE_TYPE2 = "PulseType2"
    PULSE_TYPE3 = "PulseType3"