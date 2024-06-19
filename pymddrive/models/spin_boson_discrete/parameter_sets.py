import attr
from attrs import define, field

@define
class TempelaarJCP2018:
    V: float = field(default=0.5, on_setattr=attr.setters.frozen)
    E: float = field(default=0.5, on_setattr=attr.setters.frozen)
    Omega: float = field(default=0.1, on_setattr=attr.setters.frozen)
    # lambd: float = field(default=1.0, on_setattr=attr.setters.frozen)
    lambd: float = field(default=0.25, on_setattr=attr.setters.frozen)
    kT: float = field(default=1.0, on_setattr=attr.setters.frozen)

@define
class BiasedTempelaarJCP2018:
    V: float = field(default=0.5, on_setattr=attr.setters.frozen)
    E: float = field(default=29.97894736842105, on_setattr=attr.setters.frozen)
    Omega: float = field(default=0.1, on_setattr=attr.setters.frozen)
    lambd: float = field(default=0.25, on_setattr=attr.setters.frozen)
    kT: float = field(default=1.0, on_setattr=attr.setters.frozen)

@define
class BiasedTempelaarJCP2018Pulsed:
    V: float = field(default=0.5, on_setattr=attr.setters.frozen)
    E: float = field(default=29.97894736842105, on_setattr=attr.setters.frozen)
    Omega: float = field(default=0.1, on_setattr=attr.setters.frozen)
    lambd: float = field(default=0.25, on_setattr=attr.setters.frozen)
    kT: float = field(default=1.0, on_setattr=attr.setters.frozen)
    mu_in_au: float = field(default=0.04, on_setattr=attr.setters.frozen)
    dimless2au: float = field(default=0.00095, init=False)

