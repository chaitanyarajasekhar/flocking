import numpy as np
from .agent import Agent


class Boid(Agent):
    """Boid agent"""
    config = {
        "cohesion": 0.2,
        "separation": 2,
        "alignment": 0.2,
        "obstacle_avoidance": 2,
        "goal_steering": 0.5
    }

    def _cohesion(self):
        """Boids try to fly towards the center of neighbors."""
        if not self.neighbors:
            return np.zeros(self._ndim)

        center = np.zeros(self._ndim)
        for boid in self.neighbors:
            center += boid.position
        center /= len(self.neighbors)

        return center - self.position

    def _seperation(self):
        """Boids try to keep a small distance away from other objects."""
        repel = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            distance = self.distance(neighbor)
            if distance < self.size:
                # Divergence protection.
                if distance < 0.01:
                    distance = 0.01

                repel += (self.position - neighbor.position) / \
                    distance / distance
                # No averaging taken place.
                # When two neighbors are in the same position, a stronger urge
                # to move away is assumed, despite that distancing itself from
                # one neighbor automatically eludes the other.
        return repel

    def _alignment(self):
        """Boids try to match velocity with neighboring boids."""
        # If no neighbors, no change.
        if not self.neighbors:
            return np.zeros(self._ndim)

        avg_velocity = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            avg_velocity += neighbor.velocity
        avg_velocity /= len(self.neighbors)

        return avg_velocity - self.velocity

    def _obstacle_avoidance(self):
        """Boids try to avoid obstacles."""
        # NOTE: Assume there is always enough space between obstacles
        # Find the nearest obstacle in the front.
        min_distance = np.inf
        closest = -1
        for i, obstacle in enumerate(self.obstacles):
            distance = obstacle.distance(self.position)
            if (np.dot(-obstacle.direction(self.position), self.velocity) > 0  # In the front
                    and distance < min_distance):
                closest, min_distance = i, distance

        # No obstacles in front.
        if closest < 0:
            return np.zeros(self.ndim)

        obstacle = self.obstacles[closest]
        # normal distance of obstacle to velocity, note that min_distance is obstacle's distance
        obstacle_direction = -obstacle.direction(self.position)
        sin_theta = np.linalg.norm(
            np.cross(self.direction, obstacle_direction))
        normal_distance = (min_distance + obstacle.size) * \
            sin_theta - obstacle.size
        # Decide if self is on course of collision.
        if normal_distance < self.size:
            # normal direction away from obstacle
            cos_theta = np.sqrt(1 - sin_theta * sin_theta)
            turn_direction = self.direction * cos_theta - obstacle_direction
            turn_direction = turn_direction / np.linalg.norm(turn_direction)
            # Stronger the obstrution, stronger the turn.
            return turn_direction * ((self.size - normal_distance) / max(min_distance, self.size)) ** 2

        # Return 0 if obstacle does not obstruct.
        return np.zeros(self.ndim)

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

        self._acceleration = (self.config['cohesion'] * self._cohesion() +
                              self.config['separation'] * self._seperation() +
                              self.config['alignment'] * self._alignment() +
                              self.config['obstacle_avoidance'] * self._obstacle_avoidance() +
                              self.config['goal_steering'] * goal_steering)

    @classmethod
    def set_model(cls, cohesion, separation, alignment, obstacle_avoidance, goal_steering):
        cls.config['cohesion'] = cohesion
        cls.config['separation'] = separation
        cls.config['alignment'] = alignment
        cls.config['obstacle_avoidance'] = obstacle_avoidance
        cls.config['goal_steering'] = goal_steering
