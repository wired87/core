# Injection Manager - Workflow Descriptions

This directory contains all documentation and workflow descriptions for the Injection Management package.

## Documentation Files

### 1. README.md
Complete implementation summary of the injection management package including:
- Overview and features
- BigQuery schema
- WebSocket API reference
- Testing and verification guides
- Usage examples

### 2. DATATYPES_USAGE.md
Comprehensive guide for using WebSocket dataclass objects:
- Status, InjectionData, WebSocketRequest, WebSocketResponse
- Creating and validating data objects
- Helper functions and utilities
- Complete handler examples
- Migration guide from manual dicts

### 3. SEND_STRUCTURES.md
Documentation of all WebSocket send data structures:
- All 4 handler response formats (set_inj, del_inj, get_inj_user, get_inj_list)
- Success and error response patterns
- Frontend usage examples
- Data structure patterns

## Quick Navigation

**For Users:**
- Getting Started → README.md
- API Reference → README.md (WebSocket API section)

**For Developers:**
- Data Types → DATATYPES_USAGE.md
- Response Structures → SEND_STRUCTURES.md
- Integration Guide → README.md (Integration section)

## File Organization

```
core/injection_manager/
├── workflow_descriptions/
│   ├── INDEX.md (this file)
│   ├── README.md
│   ├── DATATYPES_USAGE.md
│   └── SEND_STRUCTURES.md
├── __init__.py
├── injection.py
└── test.py
```

## Related Documentation

- **WebSocket Types**: `WEBSOCKET_TYPES.txt` (in parent directory)
- **Data Types Module**: `core/websocket_datatypes.py`
- **Test Suite**: `test.py`

## Version History

- **v1.3** (2025-12-26): Added dataclass integration
- **v1.2** (2025-12-26): Implemented status object pattern
- **v1.1** (2025-12-25): Added uniform WebSocket structure
- **v1.0** (2025-12-25): Initial implementation
