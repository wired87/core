# WebSocket Data Types - Quick Reference

## Import
```python
from core.websocket_datatypes import (
    Status, AuthData, InjectionData,
    WebSocketRequest, WebSocketResponse,
    create_injection_response, create_list_response
)
```

## Core Classes

### 1. AuthData
```python
# Create auth context
auth = AuthData.create(user_id="user_123", session_id="sess_456", env_id="env_789")
auth.to_dict()  # Excludes None values
```

### 2. Status
```python
Status.success()  # Success status
Status.error("Error message", code="ERR_001")  # Error with code
status.is_success()  # Check if success
```

### 3. InjectionData
```python
# Create injection
inj = InjectionData.create(
    id="inj_123",
    time_values=[0, 10, 20],
    energy_values=[1.0, 2.0, 3.0],
    ntype="electron"
)

# Validate and use
if inj.validate():
    manager.set_inj(inj.to_dict(), user_id)

# Access data
times = inj.get_time_values()
energies = inj.get_energy_values()
```

### 4. WebSocketRequest
```python
# Parse incoming request
request = WebSocketRequest.from_dict(payload)
auth = request.auth  # Extract auth
data = request.get_data_field("inj_object")  # Get field with default
```

### 5. WebSocketResponse
```python
# Success response
response = WebSocketResponse.success(
    type="SET_INJ",
    data={"injection_id": "inj_123"},
    auth=auth
)

# Error response
response = WebSocketResponse.error(
    type="SET_INJ",
    error="Validation failed",
    data=None,
    auth=auth
)

# Send response
await self.send(json.dumps(response.to_dict()))
```

## Helper Functions

```python
# Injection response
response = create_injection_response(success=True, injection_id="inj_123", auth=auth)

# List response
response = create_list_response(type="get_inj_user", items=injections, auth=auth)
```

## Complete Handler Example

```python
async def _handle_set_inj(self, payload: dict):
    request = WebSocketRequest.from_dict(payload)
    auth = request.auth or AuthData.create(user_id=self.user_id)
    
    inj_dict = request.get_data_field("inj_object")
    if not inj_dict:
        response = WebSocketResponse.error("Missing inj_object", auth=auth)
        await self.send(json.dumps(response.to_dict()))
        return
    
    inj = InjectionData.from_dict(inj_dict)
    if not inj.validate():
        response = WebSocketResponse.error("Invalid data", auth=auth)
        await self.send(json.dumps(response.to_dict()))
        return
    
    success = self.manager.set_inj(inj.to_dict(), auth.user_id)
    response = create_injection_response(success, inj.id, auth=auth)
    await self.send(json.dumps(response.to_dict()))
```

## Benefits
- ✅ Type-safe with IDE autocomplete
- ✅ Built-in validation
- ✅ Consistent structure across all messages
- ✅ Authentication context in every message
