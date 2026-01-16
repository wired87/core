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
        BQCore.__init__(self, dataset_id=self.DATASET_ID)
        self.qb = QBrainTableManager()
        print(f"UserManager initialized with dataset: {self.DATASET_ID}")

    def initialize_qbrain_workflow(self, uid: str, email: Optional[str] = None) -> Dict[str, Any]:
        """
        Main workflow orchestrator.

        Args:
            uid: User unique identifier
            email: User email address

        Returns:
            Dictionary containing initialization results
        """

        results = {
            "user_created": False,
            "payment_created": False,
            "errors": []
        }
        
        try:
            results["user_created"] = self._ensure_user_record(uid, email)
            results["payment_created"] = self._ensure_payment_record(uid)
        except Exception as e:
            error_msg = f"Error in initialize_qbrain_workflow: {e}"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)

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
            query = f"""
                SELECT * FROM `{self.pid}.{self.DATASET_ID}.users`
                WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("uid", "STRING", uid)
                ]
            )
            
            result = self.run_query(query, conv_to_dict=True, job_config=job_config)
            return result[0] if result else None
            
        except Exception as e:
            print(f"Error retrieving user {uid}: {e}")
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
            query = f"""
                SELECT * FROM `{self.pid}.{self.DATASET_ID}.payment`
                WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("uid", "STRING", uid)
                ]
            )
            
            result = self.run_query(query, conv_to_dict=True, job_config=job_config)
            return result[0] if result else None
            
        except Exception as e:
            print(f"Error retrieving payment record for {uid}: {e}")
            return None

    def get_standard_stack(self, user_id:str):
        self._ensure_user_record(user_id)
        query = f"""
        SELECT * from `{self.pid}.{self.DATASET_ID}.users`
        WHERE id = @user_id AND (status != 'deleted' OR status IS NULL)
        LIMIT 1
        """
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("user_id", "STRING", user_id)
            ]
        )

        result = self.run_query(
            query,
            job_config=job_config,
            conv_to_dict=True,
        )

        print("get_standard_stack RESULT:", result)
        
        if not result:
            return False

        sm_stack_status = result[0]["sm_stack_status"]

        if sm_stack_status == "created":
            return True

        return False

    def set_standard_stack(self, user_id):
        self._ensure_user_record(user_id)
        
        self.qb.set_item("users", {"sm_stack_status": "created"}, keys={"id": user_id})
        print("set_standard_stack")

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
            # Check if user already exists
            query = f"""
                SELECT id FROM `{self.pid}.{self.DATASET_ID}.users`
                WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("uid", "STRING", uid)
                ]
            )
            
            result = self.run_query(query, conv_to_dict=True, job_config=job_config)
            
            if result:
                print(f"User {uid} already exists")
                return True
            
            # Create new user record
            user_data = {
                "id": uid,
                "email": email or None,
                "status": "active"
            }
            
            print(f"Creating new user record: {user_data}")
            self.qb.set_item("users", user_data, keys={"id": uid})
            print(f"User {uid} created successfully")
            return True
            
        except Exception as e:
            print(f"Error ensuring user record: {e}")
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
            # Check if payment record already exists
            query = f"""
                SELECT id FROM `{self.pid}.{self.DATASET_ID}.payment`
                WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
                LIMIT 1
            """
            
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter("uid", "STRING", uid)
                ]
            )
            
            result = self.bqclient.query(query, job_config=job_config).result()
            
            if result.total_rows > 0:
                print(f"Payment record for user {uid} already exists")
                return True
            
            # Create new payment record (free tier)
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
            
            print(f"Creating new payment record: {payment_data}")
            self.qb.set_item("payment", payment_data, keys={"id": uid})
            print(f"Payment record for user {uid} created successfully")
            return True
            
        except Exception as e:
            print(f"Error ensuring payment record: {e}")
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
                print("No fields to update")
                return False
            
            return self.qb.set_item("payment", updates, keys={"id": uid})
            
        except Exception as e:
            print(f"Error updating payment record: {e}")
            return False
