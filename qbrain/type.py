from pydantic import BaseModel


###### NODE CFG TYPES ################################

class PhaseType(BaseModel):
    id: str
    iterations: int
    max_val_multiplier: int


class NodeCFGType(BaseModel):
    max_value: int or float or str
    phase: list[PhaseType]

######################################




