import time
from google.cloud import pubsub_v1


class GcpAdminPubSub:
    """
    Manages administrative tasks for Google Cloud Pub/Sub, such as
    creating and deleting topics and subscriptions.
    """

    def __init__(self, project_id: str):
        self.project_id = project_id
        self.admin_client = pubsub_v1.PublisherClient()
        self.subscription_client = pubsub_v1.SubscriberClient()
        print(f"GcpAdminPubSub initialized for project: {self.project_id}")

    def create_topic(self, topic_id: str):
        """Creates a new Pub/Sub topic."""
        topic_path = self.admin_client.topic_path(self.project_id, topic_id)
        try:
            topic = self.admin_client.create_topic(request={"name": topic_path})
            print(f"Topic {topic.name} created.")
        except Exception as e:
            print(f"Failed to create topic {topic_id}: {e}")

    def delete_topic(self, topic_id: str):
        """Deletes a Pub/Sub topic."""
        topic_path = self.admin_client.topic_path(self.project_id, topic_id)
        try:
            self.admin_client.delete_topic(request={"topic": topic_path})
            print(f"Topic {topic_path} deleted.")
        except Exception as e:
            print(f"Failed to delete topic {topic_id}: {e}")

    def create_subscription(self, topic_id: str, subscription_id: str):
        """Creates a new subscription for a given topic."""
        topic_path = self.admin_client.topic_path(self.project_id, topic_id)
        subscription_path = self.subscription_client.subscription_path(self.project_id, subscription_id)
        try:
            subscription = self.subscription_client.create_subscription(
                request={"name": subscription_path, "topic": topic_path}
            )
            print(f"Subscription {subscription.name} created for topic {topic_path}.")
        except Exception as e:
            print(f"Failed to create subscription {subscription_id}: {e}")

    def delete_subscription(self, subscription_id: str):
        """Deletes a Pub/Sub subscription."""
        subscription_path = self.subscription_client.subscription_path(self.project_id, subscription_id)
        try:
            self.subscription_client.delete_subscription(request={"subscription": subscription_path})
            print(f"Subscription {subscription_path} deleted.")
        except Exception as e:
            print(f"Failed to delete subscription {subscription_id}: {e}")

    def set_logging_monitoring(self, topic_id: str):
        """
        Note: Logging and monitoring are typically configured via Google Cloud's
        Cloud Logging and Cloud Monitoring services, not programmatically
        through the client libraries. This method is a placeholder to
        reflect that this is a separate, platform-level configuration.
        """
        print(f"Logging and monitoring for {topic_id} is handled by platform services.")


class PubSubManager(GcpAdminPubSub):
    """
    Handles read/write operations and other important tasks for Pub/Sub.
    Inherits administrative capabilities from GcpAdminPubSub.
    """

    def __init__(self, project_id: str):
        super().__init__(project_id)
        self.publisher = pubsub_v1.PublisherClient()
        self.subscriber = pubsub_v1.SubscriberClient()
        print("PubSubManager initialized.")

    def publish_message(self, topic_id: str, data: str):
        """Publishes a string message to a topic."""
        topic_path = self.publisher.topic_path(self.project_id, topic_id)
        data_bytes = data.encode("utf-8")
        future = self.publisher.publish(topic_path, data_bytes)
        print(f"Published message ID: {future.result()}")

    def check_for_new_messages(self, subscription_id: str, max_messages: int = 1):
        """
        Checks for new messages on a subscription without blocking for a long time.

        This method uses a short-lived pull request to see if there is any data.
        It does not acknowledge the messages, so they remain on the subscription
        for the next lookup.

        Args:
            subscription_id: The subscription ID to check.
            max_messages: The maximum number of messages to pull.

        Returns:
            A list of received messages or an empty list.
        """
        subscription_path = self.subscriber.subscription_path(self.project_id, subscription_id)
        response = self.subscriber.pull(
            request={"subscription": subscription_path, "max_messages": max_messages}
        )

        if response.received_messages:
            print(f"Found {len(response.received_messages)} new messages.")
            # We do not acknowledge the messages here so they remain for a full receive operation.
            return response.received_messages
        else:
            print("No new messages found.")
            return []

    def receive_messages(self, subscription_id: str, timeout: int = 30):
        """
        Receives messages from a subscription, processing them
        with a message handler and acknowledging them.
        """
        subscription_path = self.subscriber.subscription_path(self.project_id, subscription_id)
        print(f"Listening for messages on {subscription_path} for {timeout} seconds...")

        def callback(message: pubsub_v1.subscriber.message.Message):
            print(f"Received message: {message.data.decode('utf-8')}")
            self.handle_message(message)
            message.ack()

        streaming_pull_future = self.subscriber.subscribe(subscription_path, callback=callback)

        with self.subscriber:
            try:
                streaming_pull_future.result(timeout=timeout)
            except TimeoutError:
                streaming_pull_future.cancel()
                streaming_pull_future.result()

    def handle_message(self, message: pubsub_v1.subscriber.message.Message):
        """Placeholder for custom message processing logic."""
        print("Handling message. You can add your custom logic here.")


if __name__ == "__main__":
    PROJECT_ID = "your-gcp-project-id"
    TOPIC_ID = "my-test-topic"
    SUBSCRIPTION_ID = "my-test-subscription"

    # --- Example Usage ---
    manager = PubSubManager(PROJECT_ID)

    # 1. Admin tasks (create topic and subscription)
    # Note: Running this will try to create resources. If they already exist,
    # the calls will fail gracefully.
    manager.create_topic(TOPIC_ID)
    time.sleep(5)  # Wait for the topic to be created before creating a subscription
    manager.create_subscription(TOPIC_ID, SUBSCRIPTION_ID)

    # 2. Operational tasks (publish, check, and receive messages)
    print("\n--- Publishing a message ---")
    manager.publish_message(TOPIC_ID, "This is a message to be peeked.")
    time.sleep(5)

    print("\n--- Checking for new messages (without a full receive) ---")
    peeked_messages = manager.check_for_new_messages(SUBSCRIPTION_ID)
    if peeked_messages:
        print(f"Found new messages: {len(peeked_messages)}. They have not been acknowledged.")

    # Now, a full receive operation will fetch the message.
    print("\n--- Performing a full receive operation ---")
    manager.receive_messages(SUBSCRIPTION_ID, timeout=20)

    # 3. Admin cleanup tasks (delete subscription and topic)
    print("\n--- Cleaning up resources ---")
    manager.delete_subscription(SUBSCRIPTION_ID)
    manager.delete_topic(TOPIC_ID)
