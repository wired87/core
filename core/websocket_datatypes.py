"""
WebSocket Data Type Structures

This module defines all data types used in the WebSocket workflow.
Each data type includes:
- Clear structure definition
- Validation methods
- CRUD operation helpers
- Serialization/deserialization
- Type hints for IDE support

Ensures clean, persistent, and type-safe WebSocket communication.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


# ============================================================================
# ENUMS - Status States
# ============================================================================

class StatusState(str, Enum):
    """
    WebSocket response status states.
    
    Usage:
        - SUCCESS: Operation completed successfully
        - ERROR: Operation failed with error
        - MESSAGE: Informational message
    """
    SUCCESS = "success"
    ERROR = "error"
    MESSAGE = "message"


# ============================================================================
# STATUS OBJECT
# ============================================================================

@dataclass
class Status:
    """
    Status object for WebSocket responses.
    Contains all status-related information.
    
    Attributes:
        state: Current status state (success/error/message)
        error: Error message (only present if state is ERROR)
        message: Optional informational message
        code: Optional status code for error categorization
    
    Usage:
        # Success status
        status = Status.success()
        
        # Error status
        status = Status.error("Missing required field")
        
        # Custom status
        status = Status(state=StatusState.SUCCESS, message="Operation queued")
    """
    state: StatusState
    error: Optional[str] = None
    message: Optional[str] = None
    code: Optional[str] = None
    
    @classmethod
    def success(cls, message: Optional[str] = None) -> "Status":
        """Create a success status."""
        return cls(state=StatusState.SUCCESS, message=message)
    
    @classmethod
    def error(cls, error: str, code: Optional[str] = None) -> "Status":
        """Create an error status."""
        return cls(state=StatusState.ERROR, error=error, code=code)
    
    @classmethod
    def info(cls, message: str) -> "Status":
        """Create an informational status."""
        return cls(state=StatusState.MESSAGE, message=message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values and methods."""
        result = {}
        for k, v in asdict(self).items():
            # Skip None values and callable objects (methods)
            if v is None or callable(v):
                continue
            # Convert enum values to strings
            if isinstance(v, Enum):
                result[k] = v.value
            else:
                result[k] = v
        return result
    
    def is_success(self) -> bool:
        """Check if status represents success."""
        return self.state == StatusState.SUCCESS
    
    def is_error(self) -> bool:
        """Check if status represents error."""
        return self.state == StatusState.ERROR


# ============================================================================
# AUTHENTICATION DATA
# ============================================================================

@dataclass
class AuthData:
    """
    Authentication data for WebSocket messages.
    Contains authentication context for all operations.
    
    Attributes:
        user_id: User identifier
        session_id: Session identifier  
        env_id: Environment identifier
        timestamp: Optional timestamp of auth creation
    
    Usage:
        # Create auth data
        auth = AuthData.create(
            user_id="user_123",
            session_id="session_456",
            env_id="env_789"
        )
        
        # Create with only required fields
        auth = AuthData.create(user_id="user_123")
        
        # Convert to dict
        auth_dict = auth.to_dict()
    """
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    env_id: Optional[str] = None
    timestamp: Optional[str] = None
    
    @classmethod
    def create(cls, user_id: str, session_id: Optional[str] = None, 
               env_id: Optional[str] = None) -> "AuthData":
        """
        Create auth data with current timestamp.
        
        Args:
            user_id: User identifier (required)
            session_id: Optional session identifier
            env_id: Optional environment identifier
        
        Returns:
            AuthData instance with timestamp
        """
        return cls(
            user_id=user_id,
            session_id=session_id,
            env_id=env_id,
            timestamp=datetime.now().isoformat()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> Optional["AuthData"]:
        """
        Create AuthData from dictionary.
        
        Args:
            data: Dictionary containing auth data or None
        
        Returns:
            AuthData instance or None
        """
        if not data:
            return None
        return cls(**data)



# ============================================================================
# INJECTION DATA TYPE
# ============================================================================

@dataclass
class InjectionData:
    """
    Injection data structure for energy designer.
    
    Attributes:
        id: Unique injection identifier
        data: Two arrays - [time_values, energy_values]
        ntype: Node type (excitatory, inhibitory, modulatory, etc.)
        user_id: Owner user ID (auto-filled)
        created_at: Creation timestamp (auto-filled)
    
    Usage:
        # Create new injection
        inj = InjectionData.create(
            id="inj_123",
            time_values=[0, 10, 20],
            energy_values=[1.0, 2.0, 3.0],
            ntype="excitatory"
        )
        
        # Validate
        if inj.validate():
            db.save(inj)
        
        # Convert to dict for JSON
        data = inj.to_dict()
    """
    id: str
    data: List[List[Union[int, float]]]
    ntype: str
    user_id: Optional[str] = None
    created_at: Optional[str] = None
    
    @classmethod
    def create(cls, id: str, time_values: List[Union[int, float]], 
               energy_values: List[Union[int, float]], ntype: str,
               user_id: Optional[str] = None) -> "InjectionData":
        """
        Create injection with time and energy arrays.
        
        Args:
            id: Unique injection identifier
            time_values: Array of time values from energy designer
            energy_values: Array of energy/strength values
            ntype: Node type
            user_id: Optional user ID
        
        Returns:
            InjectionData instance
        """
        return cls(
            id=id,
            data=[time_values, energy_values],
            ntype=ntype,
            user_id=user_id,
            created_at=datetime.now().isoformat()
        )
    
    def validate(self) -> bool:
        """
        Validate injection data structure.
        
        Returns:
            True if valid, False otherwise
        
        Validation rules:
            - data must be list of 2 arrays
            - Both arrays must have same length
            - Arrays cannot be empty
            - id and ntype must be present
        """
        if not isinstance(self.data, list) or len(self.data) != 2:
            return False
        
        if not isinstance(self.data[0], list) or not isinstance(self.data[1], list):
            return False
        
        if len(self.data[0]) != len(self.data[1]):
            return False
        
        if len(self.data[0]) == 0:
            return False
        
        if not self.id or not self.ntype:
            return False
        
        return True
    
    def get_time_values(self) -> List[Union[int, float]]:
        """Get time values array."""
        return self.data[0] if self.data else []
    
    def get_energy_values(self) -> List[Union[int, float]]:
        """Get energy values array."""
        return self.data[1] if len(self.data) > 1 else []
    
    def update_data(self, time_values: List[Union[int, float]], 
                    energy_values: List[Union[int, float]]) -> None:
        """
        Update injection data arrays.
        
        Args:
            time_values: New time values
            energy_values: New energy values
        """
        self.data = [time_values, energy_values]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InjectionData":
        """
        Create InjectionData from dictionary.
        
        Args:
            data: Dictionary containing injection data
        
        Returns:
            InjectionData instance
        """
        return cls(**data)


# ============================================================================
# WEBSOCKET RESPONSE
# ============================================================================

@dataclass
class WebSocketResponse:
    """
    Standard WebSocket response structure.
    
    Attributes:
        type: Message type identifier
        status: Status object with state and error info
        data: Case-specific data (pure, no status info)
        timestamp: ISO format timestamp
        count: Optional count for list responses
    
    Usage:
        # Success response with data
        response = WebSocketResponse.success(
            type="set_inj",
            data={"injection_id": "inj_123"}
        )
        
        # Error response
        response = WebSocketResponse.error(
            type="set_inj",
            error="Missing required field"
        )
        
        # List response
        response = WebSocketResponse.success(
            type="get_inj_user",
            data=[...],
            count=5
        )
        
        # Convert to JSON
        json_data = response.to_dict()
    """
    type: str
    status: Status
    data: Any
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    auth: Optional[AuthData] = None
    count: Optional[int] = None
    
    @classmethod
    def success(cls, type: str, data: Any, count: Optional[int] = None,
                auth: Optional[AuthData] = None) -> dict:
        """
        Create success response.
        
        Args:
            type: Message type
            data: Response data
            count: Optional count for list responses
            auth: Optional authentication data
        
        Returns:
            WebSocketResponse with success status
        """
        return dict(
            type=type,
            status=Status.success(),
            data=data,
            count=count,
            auth=auth
        )

    @classmethod
    def error(cls, type: str, error: str, code: Optional[str] = None,
              data: Any = None, auth: Optional[AuthData or dict] = None) -> dict:
        """
        Create error response.
        
        Args:
            type: Message type
            error: Error message
            code: Optional error code
            data: Optional error-specific data (defaults to None or [])
            auth: Optional authentication data
        
        Returns:
            WebSocketResponse with error status
        """
        return dict(
            type=type,
            status=Status.error(error, code),
            data=data if data is not None else None,
            auth=auth
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.
        Excludes None values except for data field.
        
        Returns:
            Dictionary ready for json.dumps()
        """
        result = {
            "type": self.type,
            "status": self.status.to_dict(),
            "data": self.data,
            "auth": self.auth,
            "timestamp": self.timestamp
        }
        
        return result
    
    def is_success(self) -> bool:
        """Check if response represents success."""
        return self.status.is_success()
    
    def is_error(self) -> bool:
        """Check if response represents error."""
        return self.status.is_error()


# ============================================================================
# WEBSOCKET REQUEST
# ============================================================================

@dataclass
class WebSocketRequest:
    """
    Standard WebSocket request structure.
    
    Attributes:
        type: Message type identifier
        data: Request-specific data
        timestamp: ISO format timestamp
    
    Usage:
        # Parse incoming request
        request = WebSocketRequest.from_dict(json.loads(text_data))
        
        # Validate request
        if request.validate():
            handle_request(request)
        
        # Create new request (for testing)
        request = WebSocketRequest.create(
            type="set_inj",
            data={"inj_object": {...}}
        )
    """
    type: str
    data: Dict[str, Any]
    timestamp: str
    auth: Optional[AuthData] = None
    
    @classmethod
    def create(cls, type: str, data: Dict[str, Any],
               auth: Optional[AuthData] = None) -> "WebSocketRequest":
        """
        Create new request with current timestamp.
        
        Args:
            type: Message type
            data: Request data
            auth: Optional authentication data
        
        Returns:
            WebSocketRequest instance
        """
        return cls(
            type=type,
            data=data,
            timestamp=datetime.now().isoformat(),
            auth=auth
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WebSocketRequest":
        """
        Create request from dictionary.
        
        Args:
            data: Dictionary from JSON parse
        
        Returns:
            WebSocketRequest instance
        """
        auth_data = data.get("auth")
        auth = AuthData.from_dict(auth_data) if auth_data else None
        
        # Auto-generate timestamp if not provided by frontend
        timestamp = data.get("timestamp", datetime.now().isoformat())
        
        return cls(
            type=data["type"],
            data=data["data"],
            timestamp=timestamp,
            auth=auth
        )
    
    def validate(self) -> bool:
        """
        Validate request structure.
        
        Returns:
            True if valid, False otherwise
        """
        return bool(self.type and isinstance(self.data, dict))
    
    def get_data_field(self, field: str, default: Any = None) -> Any:
        """
        Safely get field from data.
        
        Args:
            field: Field name
            default: Default value if not found
        
        Returns:
            Field value or default
        """
        return self.data.get(field, default)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_injection_response(success: bool, injection_id: Optional[str] = None,
                              error: Optional[str] = None,
                              auth: Optional[AuthData] = None) -> dict:
    """
    Helper to create injection operation response.
    
    Args:
        success: Operation success status
        injection_id: Injection ID if successful
        error: Error message if failed
        auth: Optional authentication data
    
    Returns:
        WebSocketResponse for injection operation
    
    Usage:
        # Success
        response = create_injection_response(True, "inj_123", auth=auth)
        
        # Error
        response = create_injection_response(False, error="Validation failed", auth=auth)
    """
    if success:
        return dict(
            type="set_inj",
            data={"injection_id": injection_id},
            auth=auth
        )
    else:
        return dict(
            type="set_inj",
            error=error or "Operation failed",
            data=None,
            auth=auth
        )


def create_list_response(type: str, items: List[Any],
                         error: Optional[str] = None,
                         auth: Optional[AuthData or dict] = None) -> dict:
    """
    Helper to create list response.
    
    Args:
        type: Message type (get_inj_user, get_inj_list, etc.)
        items: List of items
        error: Error message if failed
        auth: Optional authentication data
    
    Returns:
        WebSocketResponse for list operation
    
    Usage:
        # Success
        response = create_list_response("get_inj_user", injections, auth=auth)
        
        # Error
        response = create_list_response("get_inj_user", [], error="Not found", auth=auth)
    """
    if error:
        response = dict(
            type=type,
            error=error,
            data=[],
            auth=auth
        )
    else:
        response = dict(
            type=type,
            data=items,
            count=len(items),
            auth=auth
        )
    print("Websocket response created:", response)
    return response


