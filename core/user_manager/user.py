"""
User Management Package with BigQuery Integration

Handles dataset and table management for the QBRAIN ecosystem including:
- Users, payment records, injections, environments, metadata, and modules
"""

from datetime import datetime
from typing import Optional, Dict, Any
from google.cloud import bigquery

from a_b_c.bq_agent._bq_core.bq_handler import BQCore


from core.qbrain_manager import QBrainTableManager

_USER_DEBUG = "[UserManager]"


class UserManager(BQCore):
    """
    Manages user data and records in BigQuery.
    Extends BQCore to leverage existing BigQuery functionality.
    
    Note: Table creation is handled by QBrainTableManager at server startup.
    """

    DATASET_ID = "QBRAIN"
    TABLES = {
        "users": "users",
        "payment": "payment",
        "injections": "injections",
        "environments": "environments",
        "metadata": "metadata",
        "modules": "modules"
    }
    
    # Class-level cache to verify tables only once per server process
    _tables_verified = False

    def __init__(self):
        """Initialize UserManager with QBRAIN dataset."""
        try:
            BQCore.__init__(self, dataset_id=self.DATASET_ID)
            self.qb = QBrainTableManager()
            print(f"{_USER_DEBUG} initialized with dataset: {self.DATASET_ID}")
        except Exception as e:
            print(f"{_USER_DEBUG} __init__ error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def initialize_qbrain_workflow(self, uid: str, email: Optional[str] = None) -> Dict[str, Any]:
        """
        Main workflow orchestrator.

        Args:
            uid: User unique identifier
            email: User email address

        Returns:
            Dictionary containing initialization results
        """
        results = {"user_created": False, "errors": []}
        try:
            print(f"{_USER_DEBUG} initialize_qbrain_workflow: uid={uid}")
            results["user_created"] = self._ensure_user_record(uid, email)
            print(f"{_USER_DEBUG} initialize_qbrain_workflow: done, user_created={results['user_created']}")
        except Exception as e:
            error_msg = f"Error in initialize_qbrain_workflow: {e}"
            print(f"{_USER_DEBUG} initialize_qbrain_workflow: error: {error_msg}")
            results["errors"].append(error_msg)
            import traceback
            traceback.print_exc()
        return results

    def get_user(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve user record.
        
        Args:
            uid: User unique identifier
            
        Returns:
            User record dictionary or None if not found
        """
        try:
            print(f"{_USER_DEBUG} get_user: uid={uid}")
            query = f"""
                SELECT * FROM `{self.pid}.{self.DATASET_ID}.users`
                WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
            )
            result = self.run_query(query, conv_to_dict=True, job_config=job_config)
            print(f"{_USER_DEBUG} get_user: found={bool(result)}")
            return result[0] if result else None
        except Exception as e:
            print(f"{_USER_DEBUG} get_user: error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_payment_record(self, uid: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve payment record for a user.
        
        Args:
            uid: User unique identifier
            
        Returns:
            Payment record dictionary or None if not found
        """
        try:
            print(f"{_USER_DEBUG} get_payment_record: uid={uid}")
            query = f"""
                SELECT * FROM `{self.pid}.{self.DATASET_ID}.payment`
                WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
            )
            result = self.run_query(query, conv_to_dict=True, job_config=job_config)
            print(f"{_USER_DEBUG} get_payment_record: found={bool(result)}")
            return result[0] if result else None
        except Exception as e:
            print(f"{_USER_DEBUG} get_payment_record: error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_standard_stack(self, user_id: str):
        try:
            print(f"{_USER_DEBUG} get_standard_stack: user_id={user_id}")
            self._ensure_user_record(user_id)
            query = f"""
                SELECT * from `{self.pid}.{self.DATASET_ID}.users`
                WHERE id = @user_id AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
            )
            result = self.run_query(query, job_config=job_config, conv_to_dict=True)
            if not result:
                print(f"{_USER_DEBUG} get_standard_stack: no result")
                return False
            sm_stack_status = result[0].get("sm_stack_status")
            ok = sm_stack_status == "created"
            print(f"{_USER_DEBUG} get_standard_stack: sm_stack_status={sm_stack_status}, ok={ok}")
            return ok
        except Exception as e:
            print(f"{_USER_DEBUG} get_standard_stack: error: {e}")
            import traceback
            traceback.print_exc()
            return False

    def set_standard_stack(self, user_id):
        try:
            print(f"{_USER_DEBUG} set_standard_stack: user_id={user_id}")
            self._ensure_user_record(user_id)
            self.qb.set_item("users", {"sm_stack_status": "created"}, keys={"id": user_id})
            print(f"{_USER_DEBUG} set_standard_stack: done")
        except Exception as e:
            print(f"{_USER_DEBUG} set_standard_stack: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _ensure_user_record(self, uid: str, email: Optional[str] = None) -> bool:
        """
        Create user record if it doesn't exist, or verify existing record.
        
        Args:
            uid: User unique identifier
            email: User email address
            
        Returns:
            True if user record was created or already exists
        """
        try:
            print(f"{_USER_DEBUG} _ensure_user_record: uid={uid}")
            query = f"""
                SELECT id FROM `{self.pid}.{self.DATASET_ID}.users`
                WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
            )
            result = self.run_query(query, conv_to_dict=True, job_config=job_config)
            if result:
                print(f"{_USER_DEBUG} _ensure_user_record: user already exists")
                return True
            user_data = {"id": uid, "email": email or None, "status": "active"}
            print(f"{_USER_DEBUG} _ensure_user_record: creating user")
            self.qb.set_item("users", user_data, keys={"id": uid})
            print(f"{_USER_DEBUG} _ensure_user_record: created")
            return True
        except Exception as e:
            print(f"{_USER_DEBUG} _ensure_user_record: error: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _ensure_payment_record(self, uid: str) -> bool:
        """
        Create free payment record for user if it doesn't exist.
        
        Args:
            uid: User unique identifier
            
        Returns:
            True if payment record was created or already exists
        """
        try:
            print(f"{_USER_DEBUG} _ensure_payment_record: uid={uid}")
            query = f"""
                SELECT id FROM `{self.pid}.{self.DATASET_ID}.payment`
                WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            job_config = bigquery.QueryJobConfig(
                query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
            )
            result = self.bqclient.query(query, job_config=job_config).result()
            if result.total_rows > 0:
                print(f"{_USER_DEBUG} _ensure_payment_record: already exists")
                return True
            from utils.id_gen import generate_id
            payment_data = {
                "id": generate_id(),
                "uid": uid,
                "payment_type": "free",
                "stripe_customer_id": None,
                "stripe_subscription_id": None,
                "stripe_payment_intent_id": None,
                "stripe_payment_method_id": None
            }
            self.qb.set_item("payment", payment_data, keys={"id": uid})
            print(f"{_USER_DEBUG} _ensure_payment_record: created")
            return True
        except Exception as e:
            print(f"{_USER_DEBUG} _ensure_payment_record: error: {e}")
            import traceback
            traceback.print_exc()
            raise


    def update_payment_stripe_info(
        self,
        uid: str,
        stripe_customer_id: Optional[str] = None,
        stripe_subscription_id: Optional[str] = None,
        stripe_payment_intent_id: Optional[str] = None,
        stripe_payment_method_id: Optional[str] = None,
        payment_type: Optional[str] = None
    ) -> bool:
        """
        Update Stripe payment information for a user.
        
        Args:
            uid: User unique identifier
            stripe_customer_id: Stripe customer ID
            stripe_subscription_id: Stripe subscription ID
            stripe_payment_intent_id: Stripe payment intent ID
            stripe_payment_method_id: Stripe payment method ID
            payment_type: Payment type (e.g., "free", "premium", "enterprise")
            
        Returns:
            True if update was successful
        """
        try:
            print(f"{_USER_DEBUG} update_payment_stripe_info: uid={uid}")
            updates = {}
            if stripe_customer_id is not None:
                updates["stripe_customer_id"] = stripe_customer_id
            if stripe_subscription_id is not None:
                updates["stripe_subscription_id"] = stripe_subscription_id
            if stripe_payment_intent_id is not None:
                updates["stripe_payment_intent_id"] = stripe_payment_intent_id
            if stripe_payment_method_id is not None:
                updates["stripe_payment_method_id"] = stripe_payment_method_id
            if payment_type is not None:
                updates["payment_type"] = payment_type
            if not updates:
                print(f"{_USER_DEBUG} update_payment_stripe_info: no fields to update")
                return False
            out = self.qb.set_item("payment", updates, keys={"id": uid})
            print(f"{_USER_DEBUG} update_payment_stripe_info: done")
            return out
        except Exception as e:
            print(f"{_USER_DEBUG} update_payment_stripe_info: error: {e}")
            import traceback
            traceback.print_exc()
            return False
