"""
Session Manager - Quick Demo

Demonstrates the basic functionality of the SessionManager package.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.session_manager import SessionManager
from core.user_manager import UserManager


def demo():
    """Run a simple demo of session management"""
    
    print("="*60)
    print("SESSION MANAGER - QUICK DEMO")
    print("="*60)
    
    # Step 1: Initialize managers
    print("\n1. Initializing UserManager and SessionManager...")
    user_manager = UserManager()
    session_manager = SessionManager()
    
    # Step 2: Create or verify test user
    print("\n2. Creating test user...")
    test_user_id = "demo_user_12345"
    user_manager.initialize_qbrain_workflow(
        uid=test_user_id,
        email="demo@example.com"
    )
    
    # Step 3: Create a session
    print("\n3. Creating session for user...")
    session_id = session_manager.create_session(test_user_id)
    print(f"   ✓ Session created: {session_id}")
    
    # Step 4: Retrieve the session
    print("\n4. Retrieving session details...")
    session = session_manager.get_session(session_id)
    if session:
        print(f"   - Session ID: {session['id']}")
        print(f"   - User ID: {session['user_id']}")
        print(f"   - Created: {session['created_at']}")
        print(f"   - Is Active: {session['is_active']}")
    
    # Step 5: Get all user sessions
    print("\n5. Retrieving all sessions for user...")
    all_sessions = session_manager.get_user_sessions(test_user_id)
    print(f"   ✓ Found {len(all_sessions)} session(s)")
    
    # Step 6: Deactivate session
    print("\n6. Deactivating session...")
    session_manager.deactivate_session(session_id)
    
    # Verify deactivation
    session = session_manager.get_session(session_id)
    print(f"   ✓ Session is_active: {session['is_active']}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    demo()
