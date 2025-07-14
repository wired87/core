import time


class TimeHandler:


    def __init__(self):

        pass


    def state_handler(
            self,
    duration_ms,
            timestamp,
            states,
            current_state
    ):
        if isinstance(duration_ms, (int, float)):
            if (time.time() - (timestamp + float(duration_ms / 1000.0))) > 0:
               #print("State not finished yet")
                return None
            else:
                return self.update_state(
                    states,
                    current_state
                )


    def update_state(
            self,
            states,
            current_state
    ):

        for i, item in enumerate(states):
            if item == current_state:
                if i + 1 < len(states):
                    return states[i + 1]
                else:
                    return states[0]  # Optional: loop back to "rest"
