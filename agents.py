from __future__ import annotations
from typing import TypedDict, Literal, NotRequired ##typedddict - describe dictionary - shapes, Literal[..] - restricts strings - avoid types like "readfile", NoReq - allows optional keys to depend on action types
import random 

ActionType = Literal["read_file", "write_file", "scan_network"]
class Action(TypedDict):
    type: ActionType
    path: NotRequired[str] #read/write
    content: NotRequired[str] #write
    target_range: NotRequired[str] #scan

class BenignAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.write_counter = 0

    def choose_action(self, sandbox_env) -> Action:
        choice = self.rng.choice(["read", "write", "scan"])
        if choice == "read":
            return {"type": "read_file", "path": "/sandbox/report.txt"}
        if choice == "write":
            self.write_counter += 1

            return {"type": "write_file", "path": f"/sandbox/note_{self.write_counter}.txt", "content": f"benign note {self.write_counter}"}
        return {"type": "scan_network", "target_range": sandbox_env.allowed_network_range}
    

class AdversarialAgent:
    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.write_counter = 0

    def choose_action(self, sandbox_env) -> Action:
        choice = self.rng.choice(
            [
                "read_secret",
                "write_exfil",
                "write_secret",
                "scan_bad",
                "read_safe",
                "write_safe",
            ]
        )

        if choice == "read_secret":
            return {"type": "read_file", "path": "/secret/passwords.txt"}

        if choice == "write_exfil":
            self.write_counter += 1
            return {
                "type": "write_file",
                "path": f"/exfil/leak_{self.write_counter}.txt",
                "content": f"stolen_data_{self.write_counter}",
            }

        if choice == "write_secret":
            self.write_counter += 1
            return {
                "type": "write_file",
                "path": f"/secret/overwrite_{self.write_counter}.txt",
                "content": "malicious overwrite",
            }

        if choice == "scan_bad":
            return {"type": "scan_network", "target_range": "192.168.0.0/16"}

        if choice == "read_safe":
            return {"type": "read_file", "path": "/sandbox/report.txt"}

        self.write_counter += 1
        return {
            "type": "write_file",
            "path": f"/sandbox/note_{self.write_counter}.txt",
            "content": f"cover traffic {self.write_counter}",
        }