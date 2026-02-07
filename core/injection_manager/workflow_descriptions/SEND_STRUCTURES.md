# All WebSocket self.send Data Structures - Injection Handlers

## Injection Manager WebSocket Responses

### 1. SET_INJ Handler (_handle_set_inj)

**Success Response:**
```json
{
    "type": "SET_INJ",
    "data": {
        "success": true,
        "injection_id": "inj_123"
    },
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

**Error - Missing inj_object:**
```json
{
    "type": "SET_INJ",
    "data": {
        "success": false,
        "error": "Missing inj_object"
    },
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

**Error - Exception:**
```json
{
    "type": "SET_INJ",
    "data": {
        "success": false,
        "error": "Exception message"
    },
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

---

### 2. DEL_INJ Handler (_handle_del_inj)

**Success Response:**
```json
{
    "type": "del_inj",
    "data": {
        "success": true,
        "injection_id": "inj_123"
    },
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

**Error - Missing injection_id:**
```json
{
    "type": "del_inj",
    "data": {
        "success": false,
        "error": "Missing injection_id"
    },
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

**Error - Exception:**
```json
{
    "type": "del_inj",
    "data": {
        "success": false,
        "error": "Exception message"
    },
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

---

### 3. GET_INJ_USER Handler (_handle_get_inj_user)

**Success Response:**
```json
{
    "type": "get_inj_user",
    "data": [
        {
            "id": "inj_1",
            "user_id": "user_123",
            "data": [[0, 10, 20], [1.0, 2.0, 3.0]],
            "ntype": "excitatory",
            "created_at": "2025-12-25T19:30:00"
        }
    ],
    "count": 1,
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

**Error - Exception:**
```json
{
    "type": "get_inj_user",
    "data": [],
    "error": "Exception message",
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

---

### 4. GET_INJ_LIST Handler (_handle_get_inj_list)

**Success Response:**
```json
{
    "type": "get_inj_list",
    "data": [
        {
            "id": "inj_1",
            "user_id": "user_123",
            "data": [[times], [energies]],
            "ntype": "excitatory",
            "created_at": "2025-12-25T19:30:00"
        }
    ],
    "count": 3,
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

**Error - Invalid inj_ids:**
```json
{
    "type": "get_inj_list",
    "data": [],
    "error": "inj_ids must be a list",
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

**Error - Exception:**
```json
{
    "type": "get_inj_list",
    "data": [],
    "error": "Exception message",
    "timestamp": "2025-12-25T20:00:00.123456"
}
```

---

## Data Structure Patterns

### Pattern 1: Operation Response (set_inj, del_inj)
```python
{
    "type": str,           # Message type (same as request)
    "data": {
        "success": bool,   # Operation success status
        "injection_id": str,  # ID of affected injection (if success)
        "error": str       # Error message (if failure)
    },
    "timestamp": str       # ISO format timestamp
}
```

### Pattern 2: List Response (get_inj_user, get_inj_list)
```python
{
    "type": str,           # Message type (same as request)
    "data": list,          # Array of injection objects (flattened)
    "count": int,          # Number of injections
    "error": str,          # Error message (if failure, optional)
    "timestamp": str       # ISO format timestamp
}
```

---

## Key Characteristics

1. **Type Matching**: Response type always matches request type
2. **Timestamp**: All responses include UTC ISO timestamp
3. **Flattened Data**: For lists, data field contains array directly (not nested)
4. **Error Handling**: Errors include descriptive message
5. **Success Field**: Operation responses include success boolean
6. **Count Field**: List responses include count at top level

---

## Frontend Usage Pattern

```javascript
ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    
    // Pattern 1: Operation responses (set_inj, del_inj)
    if (msg.data.hasOwnProperty('success')) {
        if (msg.data.success) {
            console.log('Success:', msg.data.injection_id);
        } else {
            console.error('Error:', msg.data.error);
        }
    }
    
    // Pattern 2: List responses (get_inj_user, get_inj_list)
    if (Array.isArray(msg.data)) {
        if (msg.error) {
            console.error('Error:', msg.error);
        } else {
            console.log(`Received ${msg.count} injections`);
            msg.data.forEach(inj => {
                console.log(inj.id, inj.ntype);
            });
        }
    }
};
```
