from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class FieldData(BaseModel):
    """The actual content of a single field extracted from the math/paper."""
    id: Optional[str] = Field(None, description="The unique ID for the field, often a random 6-digit string if not provided.")
    name: Optional[str] = Field(None, description="The display name or symbol identifier of the field.")
    params: List[str] = Field(default_factory=list, description="List of parameter IDs or names associated with this field.")
    equation: Optional[str] = Field(None, description="The LaTeX or math string defining the field's behavior.")
    description: Optional[str] = Field(None, description="Contextual description inferred from the paper.")
    axis_def: Optional[List[Any]] = Field(None, description="Definitions for the JAX axes/dimensions.")

class AuthContext(BaseModel):
    """Authentication metadata for the request."""
    user_id: str = Field(..., description="The user owning the field.")

class SetFieldItem(BaseModel):
    """The wrapper structure for an individual SET_FIELD operation."""
    data: List[FieldData]
