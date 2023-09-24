import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

class Agent:
    DEFAULT_WIDTH = 200
    DEFAULT_HEIGHT = 100

    def __init__(self, name, width=None, height=None):
        self.name = name
        self.width = width if width else Agent.DEFAULT_WIDTH
        self.height = height if height else Agent.DEFAULT_HEIGHT
        self.x = 0
        self.y = 0
        self.connections = set()

    @property
    def position(self):
        return self.x, self.y

    @position.setter
    def position(self, coordinates):
        self.x, self.y = coordinates

    def add_connection(self, agent):
        if not self.connections:
            self.connections.add(agent)

class Canvas:
    def __init__(self, width=2000, height=2000, padding=20):
        self.width = width
        self.height = height
        self.padding = padding
        self.agents = set()

    def add_agent(self, agent):
        self.agents.add(agent)

    def does_overlap(self, agent1, agent2):
        return not (agent1.x + agent1.width < agent2.x or agent1.x > agent2.x + agent2.width or
                    agent1.y + agent1.height < agent2.y or agent1.y > agent2.y + agent2.height)

    def compute_repulsion(self, agent1, agent2):
        dx = agent1.x - agent2.x
        dy = agent1.y - agent2.y
        distance = max(1.0, (dx ** 2 + dy ** 2) ** 0.5)
        force = 100.0 / distance ** 2
        return dx * force / distance, dy * force / distance

    def compute_attraction(self, agent1, agent2):
        dx = agent1.x - agent2.x
        dy = agent1.y - agent2.y
        distance = max(0.01, (dx ** 2 + dy ** 2) ** 0.5)
        force = distance ** 2 / 100.0
        return dx * force / distance, dy * force / distance

    def initialize_grid(self):
        row_size = int(self.width / (Agent.DEFAULT_WIDTH + self.padding))
        for index, agent in enumerate(self.agents):
            row = index // row_size
            col = index % row_size
            agent.position = (col * (agent.width + self.padding), row * (agent.height + self.padding))

    def boundary_repulsion(self, agent):
        force_x = 0
        force_y = 0
        epsilon = 0.01

        if agent.x < self.width * 0.1:
            force_x = 100 / (agent.x + epsilon)
        elif agent.x > self.width * 0.9:
            force_x = -100 / (self.width - agent.x + epsilon)
        
        if agent.y < self.height * 0.1:
            force_y = 100 / (agent.y + epsilon)
        elif agent.y > self.height * 0.9:
            force_y = -100 / (self.height - agent.y + epsilon)
        
        return force_x, force_y

    def auto_layout(self, iterations=500):
        self.initialize_grid()

        all_agents = set(self.agents)
        for agent in self.agents:
            all_agents.update(agent.connections)

        damping = 0.5
        damping_decay = 0.99
        
        for _ in range(iterations):
            forces = {agent: [0, 0] for agent in all_agents}

            for agent1 in self.agents:
                for agent2 in self.agents:
                    if agent1 != agent2:
                        if agent2 in agent1.connections or agent1 in agent2.connections:
                            fx, fy = self.compute_attraction(agent1, agent2)
                        else:
                            fx, fy = self.compute_repulsion(agent1, agent2)
                        forces[agent1][0] += fx
                        forces[agent1][1] += fy

            for agent in self.agents:
                fx, fy = self.boundary_repulsion(agent)
                forces[agent][0] += fx
                forces[agent][1] += fy

            for agent, (fx, fy) in forces.items():
                agent.x += fx * damping
                agent.y += fy * damping
                agent.x = min(max(agent.x, 0), self.width - agent.width)
                agent.y = min(max(agent.y, 0), self.height - agent.height)
            
            damping *= damping_decay

    def visualize(self):
        fig, ax = plt.subplots()

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal', adjustable='box')
        ax.invert_yaxis()

        for agent in self.agents:
            rect = patches.Rectangle((agent.x, agent.y), agent.width, agent.height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            plt.text(agent.x + agent.width / 2, agent.y + agent.height / 2, agent.name, ha='center', va='center', fontsize=6)

            for connection in agent.connections:
                plt.plot([agent.x + agent.width, connection.x], [agent.y + agent.height / 2, connection.y + connection.height / 2], color='blue')

        plt.show()

if __name__ == "__main__":
    canvas = Canvas()

    agents = [Agent(f"A{i}") for i in range(150)]

    for i, agent in enumerate(agents):
        if i < len(agents) - 1:
            agent.add_connection(agents[i + 1])

    canvas.agents = set(agents)
    canvas.auto_layout()
    canvas.visualize()
