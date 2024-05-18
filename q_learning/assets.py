from time import sleep

from prettytable import PrettyTable as pt


class GridWorld:
    def __init__(
        self,
        grid_size_len,
        grid_size_width,
        step_reward,
        cliff_penalty,
        start_state,
        terminal_state,
        cliff_states,
    ):
        self.grid_size_len = grid_size_len
        self.grid_size_width = grid_size_width
        self.step_reward = step_reward
        self.cliff_penalty = cliff_penalty
        self.start_state = start_state
        self.current_state = list(start_state)
        self.terminal_state = terminal_state
        self.cliff_states = cliff_states
        self.action_space = ["↑", "→", "↓", "←"]
        self.observation_space = [
            [i, j] for i in range(grid_size_width) for j in range(grid_size_len)
        ]

    def reset(self):
        self.current_state = list(self.start_state)
        return list(self.current_state)

    def step(self, action):
        if action == "↑":
            # North
            if self.current_state[0] != 0:
                self.current_state[0] -= 1
        elif action == "→":
            # East
            if self.current_state[1] != self.grid_size_len - 1:
                self.current_state[1] += 1
        elif action == "↓":
            # South
            if self.current_state[0] != self.grid_size_width - 1:
                self.current_state[0] += 1
        else:
            # West
            if self.current_state[1] != 0:
                self.current_state[1] -= 1
        if self.current_state in self.cliff_states:
            return (list(self.current_state), self.cliff_penalty, True)
        elif self.current_state == self.terminal_state:
            return (list(self.current_state), 0, True)
        else:
            return (list(self.current_state), self.step_reward, False)

    def render(self):
        table = pt()
        table.field_names = [str(i) for i in range(self.grid_size_len)]
        for i in range(self.grid_size_width):
            row = []
            for j in range(self.grid_size_len):
                if [i, j] == self.current_state:
                    row.append("X")
                elif [i, j] in self.cliff_states:
                    row.append("C")
                elif [i, j] == self.terminal_state:
                    row.append("T")
                elif [i, j] == self.start_state:
                    row.append("S")
                else:
                    row.append(" ")
            table.add_row(row)
        print(table)
        sleep(1)


env = GridWorld(
    grid_size_len=12,
    grid_size_width=4,
    step_reward=-1,
    cliff_penalty=-100,
    start_state=[3, 0],
    terminal_state=[3, 11],
    cliff_states=[[3, i] for i in range(1, 11)],
)

params = {
    "lr": 0.01,
    "gamma": 0.99,
    "epsilon": 0.01,
    "num_episodes": 10000,
}
