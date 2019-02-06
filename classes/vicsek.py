import numpy as np
from .agent import Agent
from .obstacles import Obstacle

class Vicsek(Agent):
    config = {
        'tau': 1.0,
        'A': 1.0,
        'B': 2.0,
        'k': 2.0,
        'kappa': 1.0
    }

    def _interaction(self, other):
        r = self.size + other.size
        d = np.linalg.norm(self.position - other.position)

        if isinstance(other, Obstacle):
            n = other.direction(self.position)
        else:
            n = (self.position - other.position) / d

        repulsion = self.config['A'] * np.exp((r - d) / self.config['B']) * n
        friction = 0
        if r > d:
            repulsion += self.config['k'] * (r - d) * n # Body force.

            delta_v = other.velocity - self.velocity
            friction += self.config['kappa'] * (r - d) * (delta_v - np.dot(delta_v, n) * n)

        return repulsion + friction

    def _goal_seeking(self, goal):
        """Individual goal of the boid."""
        # As a simple example, suppose the boid would like to go as fast as it
        # can in the current direction when no explicit goal is present.
        if not goal:
            return self.velocity / self.speed

        # The urge to chase the goal is stronger when farther.
        offset = goal.position - self.position
        distance = np.linalg.norm(offset)
        target_speed = self.max_speed * min(1, distance / 20)
        target_velocity = target_speed * offset / distance
        return target_velocity - self.velocity

    def decide(self, goals):
        """Make decision for acceleration."""
        goal_steering = np.zeros(self.ndim)

        for goal in goals:
            goal_steering += self._goal_seeking(goal) * goal.priority

        interactions = 0
        for neighbor in self.neighbors:
            interactions += self._interaction(neighbor)

        for obstacle in self.obstacles:
            interactions += self._interaction(obstacle)

        self._acceleration[:] = interactions[:] + goal_steering[:]

    