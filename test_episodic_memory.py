# Simple episodic memory test case

class EpisodicMemory:
    def __init__(self):
        self.events = []

    def record_event(self, event):
        self.events.append(event)

    def retrieve_event(self, index):
        return self.events[index] if 0 <= index < len(self.events) else None

    def get_all_events(self):
        return self.events

# Create test scenario
memory = EpisodicMemory()

# Record some events
memory.record_event("User visited the homepage at 10:00 AM")
memory.record_event("User clicked on the login button at 10:01 AM")
memory.record_event("User submitted login form at 10:02 AM")

# Test retrieval
event1 = memory.retrieve_event(0)
event2 = memory.retrieve_event(1)
event3 = memory.retrieve_event(2)

print("Event 1:", event1)
print("Event 2:", event2)
print("Event 3:", event3)

# Test all events
events = memory.get_all_events()
print("All Events:", events)

# Test edge case - invalid index
invalid_event = memory.retrieve_event(5)
print("Invalid event index:", invalid_event)