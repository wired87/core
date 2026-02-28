# WebSocket Data Structures Reference

## Table of Contents
- [Message Format](#message-format)
- [Authentication](#authentication)
- [Status Objects](#status-objects)
- [Injection Messages](#injection-messages)
- [Data Types](#data-types)

---

## Message Format

All WebSocket messages follow a uniform structure:

```json
{
    "type": "message_type",
    "auth": {
        "user_id": "string",
        "session_id": "string (optional)",
        "env_id": "string (optional)",
        "timestamp": "ISO 8601 string"
    },
    "status": {
        "state": "success | error | message",
        "error": "string | null",
        "message": "string | null",
        "code": "string | null"
    },
    "data": "any (message-specific)",
    "timestamp": "ISO 8601 string",
    "count": "integer (optional, for list responses)"
}
```

---

## Authentication

### AuthData Structure

```typescript
interface AuthData {
    user_id?: string;
    session_id?: string;
    env_id?: string;
    timestamp?: string;  // ISO 8601 format
}
```

### Example

```json
{
    "user_id": "72b74d5214564004a3a86f441a4a112f",
    "session_id": "565367237012739",
    "env_id": "env_prod_001",
    "timestamp": "2025-12-26T14:00:00.000Z"
}
```

---

## Status Objects

### StatusState Enum

```typescript
enum StatusState {
    SUCCESS = "success",
    ERROR = "error",
    MESSAGE = "message"
}
```

### Status Structure

```typescript
interface Status {
    state: StatusState;
    error?: string | null;
    message?: string | null;
    code?: string | null;
}
```

### Examples

**Success:**
```json
{
    "state": "success",
    "error": null,
    "message": null,
    "code": null
}
```

**Error:**
```json
{
    "state": "error",
    "error": "Invalid injection data structure",
    "message": null,
    "code": "VAL_001"
}
```

**Info Message:**
```json
{
    "state": "message",
    "error": null,
    "message": "Processing your request...",
    "code": null
}
```

---

## Injection Messages

### 1. SET_INJ - Create/Update Injection

Creates a new injection with foreign key relationship to user, then returns all user injections.

**Workflow:**
1. Insert injection with `user_id` foreign key to `users.uid`
2. Query all injections for user (INNER JOIN with users table)
3. Respond with type `get_inj_user` containing updated injection list

**Request:**
```json
{
    "type": "SET_INJ",
    "auth": {
        "user_id": "72b74d5214564004a3a86f441a4a112f"
    },
    "data": {
        "id": "inj_1766752019296_3wq15vvv8",
        "data": [
            [0, 10, 20, 30, 40],          // Time values
            [1.0, 2.0, 3.0, 4.0, 5.0]     // Energy values
        ],
        "ntype": "ELECTRON"
    },
    "timestamp": "2025-12-26T14:00:00.000Z"
}
```

**Response (Success - Returns ALL User Injections):**
```json
{
    "type": "get_inj_user",
    "status": {
        "state": "success",
        "error": null
    },
    "data": [
        {
            "id": "inj_1766752019296_3wq15vvv8",
            "data": [[0, 10, 20, 30, 40], [1.0, 2.0, 3.0, 4.0, 5.0]],
            "ntype": "ELECTRON"
        },
        {
            "id": "inj_previous",
            "data": [[0, 5, 10], [1.5, 2.5, 3.5]],
            "ntype": "EXCITATORY"
        }
    ],
    "count": 2,
    "auth": {
        "user_id": "72b74d5214564004a3a86f441a4a112f",
        "timestamp": "2025-12-26T14:00:00.123Z"
    },
    "timestamp": "2025-12-26T14:00:00.123Z"
}
```

**Response (Error):**
```json
{
    "type": "SET_INJ",
    "status": {
        "state": "error",
        "error": "Invalid injection data structure"
    },
    "data": null,
    "auth": {
        "user_id": "72b74d5214564004a3a86f441a4a112f",
        "timestamp": "2025-12-26T14:00:00.123Z"
    },
    "timestamp": "2025-12-26T14:00:00.123Z"
}
```

**Database Operations:**
```sql
-- 1. Insert with foreign key
INSERT INTO injections (id, user_id, data, ntype)
VALUES ('inj_123', '72b74d...', '[[0,10],[1.0,2.0]]', 'ELECTRON');

-- 2. Query all user injections
SELECT i.id, i.data, i.ntype
FROM injections AS i
INNER JOIN users AS u ON i.user_id = u.uid
WHERE i.user_id = '72b74d...';
```

---

### 2. DEL_INJ - Delete Injection

Deletes an injection from the database, then returns all remaining user injections.

**Workflow:**
1. Delete injection from `injections` table by ID
2. Query remaining injections for user (INNER JOIN with users table)
3. Respond with type `get_inj_user` containing remaining injection list

**Request:**
```json
{
    "type": "del_inj",
    "auth": {
        "user_id": "72b74d5214564004a3a86f441a4a112f"
    },
    "data": {
        "id": "inj_1766752019296_3wq15vvv8"
    },
    "timestamp": "2025-12-26T14:00:00.000Z"
}
```

**Response (Success - Returns Remaining Injections):**
```json
{
    "type": "get_inj_user",
    "status": {
        "state": "success",
        "error": null
    },
    "data": [
        {
            "id": "inj_remaining_001",
            "data": [[0, 5, 10], [1.5, 2.5, 3.5]],
            "ntype": "EXCITATORY"
        },
        {
            "id": "inj_remaining_002",
            "data": [[0, 10, 20], [2.0, 3.0, 4.0]],
            "ntype": "ELECTRON"
        }
    ],
    "count": 2,
    "auth": {
        "user_id": "72b74d5214564004a3a86f441a4a112f",
        "timestamp": "2025-12-26T14:00:00.123Z"
    },
    "timestamp": "2025-12-26T14:00:00.123Z"
}
```

**Response (Error):**
```json
{
    "type": "del_inj",
    "status": {
        "state": "error",
        "error": "Failed to delete injection from database"
    },
    "data": null,
    "auth": {
        "user_id": "72b74d5214564004a3a86f441a4a112f"
    },
    "timestamp": "2025-12-26T14:00:00.123Z"
}
```

**Database Operations:**
```sql
-- 1. Delete injection
DELETE FROM injections
WHERE id = 'inj_1766752019296_3wq15vvv8';

-- 2. Query remaining user injections
SELECT i.id, i.data, i.ntype
FROM injections AS i
INNER JOIN users AS u ON i.user_id = u.uid
WHERE i.user_id = '72b74d...';
```

---

### 3. GET_INJ_USER - Get All User Injections

**Request:**
```json
{
    "type": "get_inj_user",
    "auth": {
        "user_id": "72b74d5214564004a3a86f441a4a112f"
    },
    "data": {},
    "timestamp": "2025-12-26T14:00:00.000Z"
}
```

**Response (Success):**
```json
{
    "type": "get_inj_user",
    "status": {
        "state": "success",
        "error": null
    },
    "data": [
        {
            "id": "inj_001",
            "data": [[0, 10, 20], [1.0, 2.0, 3.0]],
            "ntype": "ELECTRON"
        },
        {
            "id": "inj_002",
            "data": [[0, 5, 10], [2.0, 3.0, 4.0]],
            "ntype": "EXCITATORY"
        }
    ],
    "count": 2,
    "auth": {
        "user_id": "72b74d5214564004a3a86f441a4a112f",
        "timestamp": "2025-12-26T14:00:00.123Z"
    },
    "timestamp": "2025-12-26T14:00:00.123Z"
}
```

---

### 4. GET_INJ_LIST - Get Specific Injections

**Request:**
```json
{
    "type": "get_inj_list",
    "auth": {
        "user_id": "72b74d5214564004a3a86f441a4a112f"
    },
    "data": {
        "inj_ids": ["inj_001", "inj_002", "inj_003"]
    },
    "timestamp": "2025-12-26T14:00:00.000Z"
}
```

**Response (Success):**
```json
{
    "type": "get_inj_list",
    "status": {
        "state": "success",
        "error": null
    },
    "data": [
        {
            "id": "inj_001",
            "data": [[0, 10, 20], [1.0, 2.0, 3.0]],
            "ntype": "ELECTRON"
        },
        {
            "id": "inj_002",
            "data": [[0, 5, 10], [2.0, 3.0, 4.0]],
            "ntype": "EXCITATORY"
        }
    ],
    "count": 2,
    "auth": {
        "user_id": "72b74d5214564004a3a86f441a4a112f",
        "timestamp": "2025-12-26T14:00:00.123Z"
    },
    "timestamp": "2025-12-26T14:00:00.123Z"
}
```

---

## Data Types

### InjectionData Structure

```typescript
interface InjectionData {
    id: string;
    data: number[][];  // [[time_values], [energy_values]]
    ntype: string;     // Node type: ELECTRON, EXCITATORY, etc.
}
```

### Storage Format

**In Application Memory:**
```json
{
    "id": "inj_123",
    "data": [[0, 10, 20, 30], [1.0, 2.0, 3.0, 4.0]],
    "ntype": "ELECTRON"
}
```

**In BigQuery (JSON field):**
```json
{
    "id": "inj_123",
    "data": "[[0,10,20,30],[1.0,2.0,3.0,4.0]]",  // JSON string
    "ntype": "ELECTRON"
}
```

### Validation Rules

1. **`id`**: Required, string, unique identifier
2. **`data`**: Required, array of exactly 2 arrays
   - `data[0]`: Time values (array of numbers)
   - `data[1]`: Energy values (array of numbers)
   - Both arrays must have the same length
3. **`ntype`**: Required, string (node type identifier)

### Example Valid Injection

```json
{
    "id": "inj_1766752019296",
    "data": [
        [0, 8, 92],      // Time: 0ms, 8ms, 92ms
        [25, 25, 0]      // Energy: 25, 25, 0
    ],
    "ntype": "ELECTRON"
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| `VAL_001` | Invalid injection data structure |
| `VAL_002` | Missing required field |
| `DB_001` | Database connection error |
| `AUTH_001` | Authentication failed |
| `NOT_FOUND` | Resource not found |

---

## WebSocket Connection

### Connection URL

```
ws://localhost:8000/run/?user_id={USER_ID}
```

### Example Connection (JavaScript)

```javascript
const userId = "72b74d5214564004a3a86f441a4a112f";
const ws = new WebSocket(`ws://localhost:8000/run/?user_id=${userId}`);

ws.onopen = () => {
    console.log('Connected to Relay Station');
};

ws.onmessage = (event) => {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
};

ws.send(JSON.stringify({
    type: 'get_inj_user',
    auth: { user_id: userId },
    data: {},
    timestamp: new Date().toISOString()
}));
```

---

## Complete Request/Response Examples

### Set Injection Workflow

1. **Client sends:**
```json
{
    "type": "SET_INJ",
    "auth": { "user_id": "user_123" },
    "data": {
        "id": "inj_new",
        "data": [[0, 10], [1.0, 2.0]],
        "ntype": "ELECTRON"
    },
    "timestamp": "2025-12-26T14:00:00.000Z"
}
```

2. **Server validates** injection structure

3. **Server stores** in BigQuery

4. **Server responds:**
```json
{
    "type": "SET_INJ",
    "status": { "state": "success", "error": null },
    "data": { "injection_id": "inj_new" },
    "auth": { "user_id": "user_123", "timestamp": "2025-12-26T14:00:00.123Z" },
    "timestamp": "2025-12-26T14:00:00.123Z"
}
```

---

## Notes

- All timestamps use ISO 8601 format: `YYYY-MM-DDTHH:mm:ss.sssZ`
- The `auth` object is optional in requests but recommended
- The `count` field appears only in list responses
- Nested arrays in `data` are stored as JSON strings in BigQuery
- Server auto-generates timestamps if not provided by client
