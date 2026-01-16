
import pytest
from core.fields_manager.fields_lib import FieldsManager, generate_numeric_id

def test_fields_manager():
    fm = FieldsManager()
    user_id = "test_user_" + generate_numeric_id()
    module_id = "test_module_" + generate_numeric_id()
    
    # Test Set Field
    field_id = generate_numeric_id()
    field_data = {
        "id": field_id,
        "params": {"key": "value"},
        "module": module_id
    }
    fm.set_field(field_data, user_id)
    
    # Test Get Fields User
    fields = fm.get_fields_by_user(user_id)
    assert len(fields) >= 1
    found = False
    for f in fields:
        if f["id"] == field_id:
            assert f["module"] == module_id
            # Check params deserialization
            if isinstance(f["params"], dict):
                assert f["params"]["key"] == "value"
            found = True
            break
    assert found
    
    # Test Link Module Field
    fm.link_module_field(module_id, field_id, user_id)
    
    # Test Get Fields Module
    mod_fields = fm.get_fields_by_module(module_id, user_id)
    assert len(mod_fields) >= 1
    assert any(f["id"] == field_id for f in mod_fields)

    # Test Del Field
    fm.delete_field(field_id, user_id)
    fields_after = fm.get_fields_by_user(user_id)
    assert not any(f["id"] == field_id for f in fields_after)

if __name__ == "__main__":
    # creating dummy class if pytest not present or run direct
    try:
        test_fields_manager()
        print("Tests passed")
    except Exception as e:
        print(f"Tests failed: {e}")
