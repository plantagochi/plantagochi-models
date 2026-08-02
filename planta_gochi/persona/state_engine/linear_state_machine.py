from planta_gochi.persona.state_engine.base import StateEngine
from planta_gochi.persona.state_engine.linear_state_enums import LinearEvents, LinearState

class LinearStateMachine(StateEngine):
    def __init__(self,
                 thresholds: list, 
                 state_prompts: dict,
                 event_prompts: dict,
                 default_dialog: dict,
                 prev_value = None):
        self.prev_value = prev_value
        self.thresholds = thresholds
        self.state_prompts = state_prompts
        self.event_prompts = event_prompts
        self.default_dialog = default_dialog

    def _cooresponding_state_no(self, value):
        for no, threshold in enumerate(self.thresholds):
            if value < threshold:
                return no + 1
        #모든 경계선을 넘음(값이 매우 큼)
        return len(self.thresholds) + 1

    def _cooresponding_state(self, value):
        state_no = self._cooresponding_state_no(value)
        return LinearState(state_no)

    def _detect_event_enum(self, curr_value):
        if self.prev_value == None:
            self.prev_value = curr_value
            return LinearEvents(0)
        
        prev_state_no = self._cooresponding_state_no(self.prev_value)
        curr_state_no = self._cooresponding_state_no(curr_value)

        self.prev_value = curr_value

        if prev_state_no == curr_state_no:
            return LinearEvents(0)
        else:
            return LinearEvents(prev_state_no * 10 + curr_state_no)


    def detect_event(self, curr_value):
        event = self._detect_event_enum(curr_value)
        state = self._cooresponding_state(curr_value)
        is_there_a_event = event.value != 0
        
        event_prompt = self.event_prompts[event]
        state_prompt = self.state_prompts[state]
        
        result = {
            "prompt": event_prompt if is_there_a_event else state_prompt,
            "default_dialog": self.default_dialog[event if is_there_a_event else state],
            "is_there_a_event": is_there_a_event,
            "event_prompt": event_prompt,
            "state_prompt": state_prompt,
            "raw_event": event,
        }

        return result
        

    
        
    