import asyncio
import websockets

# Define the server URI to connect to
SERVER_URI = "ws://127.0.0.1:8000/run/"
TEST_MESSAGE = "TEST_DATA_REQUEST"
EXPECTED_RESPONSE = "DATA_VALIDATED_SUCCESS"


async def run_server_to_server_test():
    """
    Connects to the WebSocket server, sends a test message,
    and validates the received response.
    """
    print(f"Connecting to server at {SERVER_URI}...")
    try:
        # Establish the connection
        async with websockets.connect(SERVER_URI) as websocket:

            # --- 1. Send the test request ---
            await websocket.send(TEST_MESSAGE)
            print(f"Client sent: <{TEST_MESSAGE}>")

            # --- 2. Wait for the response ---
            response = await websocket.recv()
            print(f"Client received: <{response}>")

            # --- 3. Validate the test case result ---
            if response == EXPECTED_RESPONSE:
                print("\n✅ TEST PASSED: Response matched expected validation.")
            else:
                print(f"\n❌ TEST FAILED: Expected <{EXPECTED_RESPONSE}> but got <{response}>")

    except ConnectionRefusedError:
        print(f"\n❌ CONNECTION FAILED: Server not running at {SERVER_URI}. Please start ws_server.py first.")
    except Exception as e:
        print(f"\n❌ AN UNEXPECTED ERROR OCCURRED: {e}")


if __name__ == "__main__":
    # Start the asynchronous test execution
    asyncio.run(run_server_to_server_test())


