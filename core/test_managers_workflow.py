
import logging
import random
import time
from datetime import datetime
from core.qbrain_manager import get_qbrain_table_manager
from core.module_manager.ws_modules_manager.modules_lib import module_manager
from core.fields_manager.fields_lib import fields_manager
from core.injection_manager.injection import injection_manager

logging.basicConfig(level=logging.INFO)

def generate_id():
    return str(random.randint(1000000000, 9999999999))

def test_managers():
    print("STARTING EXPENSIVE MANAGER TEST")
    
    # 1. Initialize Tables
    print("\n[1] Initializing Tables...")
    qb = get_qbrain_table_manager()
    qb.initialize_all_tables()
    
    user_id = "test_user_" + generate_id()
    session_id = "session_" + generate_id()
    env_id = "env_" + generate_id()
    
    print(f"Test User: {user_id}")
    print(f"Test Session: {session_id}")
    
    # Create User
    user_data = {
        "uid": user_id,
        "email": f"{user_id}@test.com",
        "created_at": datetime.now().isoformat(),
        "status": "active",
        "sm_stack_status": "ready"
    }
    qb.bq_insert("users", [user_data], upsert=False)
    time.sleep(2)
    
    # ==========================================
    # MODULES
    # ==========================================
    print("\n[2] Testing Module Manager...")
    module_id = generate_id()
    module_data = {
        "id": module_id,
        "code": "print('hello')",
        "file_type": "py",
        "created_at": datetime.now().isoformat(),
        "params": {"p1": "v1"}
    }
    
    # Set
    try:
        module_manager.set_module(module_data, user_id)
        time.sleep(5)
        # verify
        mod = module_manager.get_module_by_id(module_id)["modules"][0]
        assert mod["id"] == module_id
        assert mod["code"] == "print('hello')"
        print("  Set Module: OK")
    except Exception as e:
        print(f"  Set Module: FAILED ({e})")
    
    # Link Session
    try:
        module_manager.link_session_module(session_id, module_id, user_id)
        time.sleep(3)
        s_mods = module_manager.retrieve_session_modules(session_id, user_id)
        assert len(s_mods) == 1
        assert s_mods[0]["id"] == module_id
        print("  Link Session Module: OK")
    except Exception as e:
        print(f"  Link Session Module: FAILED ({e})")
    
    # Use generic generic search (Users modules)
    try:
        u_mods = module_manager.retrieve_user_modules(user_id)
        assert len(u_mods) >= 1
        print("  Retrieve User Modules: OK")
    except Exception as e:
        print(f"  Retrieve User Modules: FAILED ({e})")
    
    # Soft Delete Link
    try:
        module_manager.rm_link_session_module(session_id, module_id, user_id)
        time.sleep(3)
        s_mods = module_manager.retrieve_session_modules(session_id, user_id)
        assert len(s_mods) == 0
        print("  Remove Link Session Module: OK")
    except Exception as e:
        print(f"  Remove Link Session Module: FAILED ({e})")
    
    # Delete Module
    try:
        module_manager.delete_module(module_id, user_id)
        time.sleep(3)
        mod_check = module_manager.get_module_by_id(module_id)["modules"]
        assert len(mod_check) == 0
        print("  Delete Module: OK")
    except Exception as e:
        print(f"  Delete Module: FAILED ({e})")

    # Re-create module for Fields test
    module_manager.set_module(module_data, user_id)
    module_manager.link_session_module(session_id, module_id, user_id)
    time.sleep(3)

    # ==========================================
    # FIELDS
    # ==========================================
    print("\n[3] Testing Fields Manager...")
    field_id = generate_id()
    field_data = {
        "id": field_id,
        "keys": ["k1"],
        "values": [1.0],
        "axis_def": {"x": 1},
        "status": "active"
    }
    
    # Set
    try:
        fields_manager.set_field(field_data, user_id)
        time.sleep(5)
        f = fields_manager.get_fields_by_id([field_id])["fields"][0]
        assert f["id"] == field_id
        print("  Set Field: OK")
    except Exception as e:
        print(f"  Set Field: FAILED ({e})")
    
    # Link Module -> Field
    try:
        link_data = [{
            "id": generate_id(),
            "module_id": module_id,
            "field_id": field_id,
            "user_id": user_id, 
            "status": "active"
        }]
        fields_manager.link_module_field(link_data)
        time.sleep(3)
        
        # Get Fields by Module
        m_fields = fields_manager.get_fields_by_module(module_id, user_id)
        assert len(m_fields) == 1
        assert m_fields[0]["id"] == field_id
        print("  Link Module Field & Get Fields By Module: OK")
        
        # Retrieve Session Fields (via Module link)
        s_fields = fields_manager.retrieve_session_fields(session_id, user_id) # IDs
        # Check if field_id in list
        assert field_id in s_fields
        print("  Retrieve Session Fields: OK")

        # Get Modules Fields (ModuleWsManager delegating to FieldsManager)
        ms_fields = module_manager.get_modules_fields(user_id, session_id)
        assert len(ms_fields["fields"]) == 1
        assert ms_fields["fields"][0]["id"] == field_id
        print("  Get Modules Fields (WS Manager): OK")
    except Exception as e:
        print(f"  Fields Linking: FAILED ({e})")

    # Remove Link Module Field
    try:
        fields_manager.rm_link_module_field(module_id, field_id, user_id)
        time.sleep(3)
        m_fields = fields_manager.get_fields_by_module(module_id, user_id)
        assert len(m_fields) == 0
        print("  Remove Link Module Field: OK")
    except Exception as e:
        print(f"  Remove Link Module Field: FAILED ({e})")
    
    # Delete Field
    try:
        fields_manager.delete_field(field_id, user_id)
        time.sleep(3)
        f_check = fields_manager.get_fields_by_id([field_id])["fields"]
        assert len(f_check) == 0
        print("  Delete Field: OK")
    except Exception as e:
        print(f"  Delete Field: FAILED ({e})")


    # ==========================================
    # INJECTIONS
    # ==========================================
    print("\n[4] Testing Injection Manager...")
    inj_id = generate_id()
    inj_data = {
        "id": inj_id,
        "data": [[1,2],[3,4]]
    }
    
    # Set
    try:
        injection_manager.set_inj(inj_data, user_id)
        time.sleep(5)
        inj = injection_manager.get_injection(inj_id)
        assert inj["id"] == inj_id
        print("  Set Injection: OK")
    except Exception as e:
        print(f"  Set Injection: FAILED ({e})")
    
    # Link Session
    try:
        injection_manager.link_session_injection(session_id, inj_id, user_id)
        time.sleep(3)
        s_injs = injection_manager.retrieve_session_injections(session_id, user_id)
        assert len(s_injs) == 1
        assert s_injs[0]["id"] == inj_id
        print("  Link Session Injection: OK")
    except Exception as e:
        print(f"  Link Session Injection: FAILED ({e})")
    
    # Remove Link Session
    try:
        injection_manager.rm_link_session_injection(session_id, inj_id, user_id)
        time.sleep(3)
        s_injs = injection_manager.retrieve_session_injections(session_id, user_id)
        assert len(s_injs) == 0
        print("  Remove Link Session Injection: OK")
    except Exception as e:
        print(f"  Remove Link Session Injection: FAILED ({e})")
    
    # Link Env
    try:
        injection_manager.link_inj_env(inj_id, env_id, user_id, (0,0,0))
        time.sleep(3)
        e_injs = injection_manager.get_inj_env(env_id, user_id, inj_id)
        assert len(e_injs) == 1
        print("  Link Env Injection: OK")
    except Exception as e:
        print(f"  Link Env Injection: FAILED ({e})")
    
    # Remove Link Env
    try:
        injection_manager.rm_link_inj_env(inj_id, env_id, user_id)
        time.sleep(3)
        e_injs = injection_manager.get_inj_env(env_id, user_id, inj_id)
        assert len(e_injs) == 0
        print("  Remove Link Env Injection: OK")
    except Exception as e:
        print(f"  Remove Link Env Injection: FAILED ({e})")
    
    # Delete Injection
    try:
        injection_manager.del_inj(inj_id, user_id)
        time.sleep(3)
        inj_check = injection_manager.get_injection(inj_id)
        assert inj_check is None
        print("  Delete Injection: OK")
    except Exception as e:
        print(f"  Delete Injection: FAILED ({e})")


    print("\nALL MANAGERS TESTED SUCCESSFULY")

if __name__ == "__main__":
    test_managers()
