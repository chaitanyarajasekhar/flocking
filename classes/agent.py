import numpy as np

class Agent:
    def __init__(self, ndim=None, size=None, vision=None,
                 max_speed=None, max_acceleration=None):
        """
        Create a boid with essential attributes.
        `ndim`: dimension of the space it resides in.
        `vision`: the visual range.
        `anticipation`: range of anticipation for its own motion.
        `comfort`: distance the agent wants to keep from other objects.
        `max_speed`: max speed the agent can achieve.
        `max_acceleratoin`: max acceleration the agent can achieve.
        """
        self._ndim = ndim if ndim else 3

        self.size = float(size) if size else 0.
        self.vision = float(vision) if vision else np.inf

        # Max speed the boid can achieve.
        self.max_speed = float(max_speed) if max_speed else None
        self.max_acceleration = float(max_acceleration) if max_acceleration else None

        self.neighbors = []
        self.obstacles = []

    def initialize(self, position, velocity):
        """Initialize agent's spactial state."""
        self._position = np.zeros(self._ndim)
        self._velocity = np.zeros(self._ndim)
        self._acceleration = np.zeros(self._ndim)

        self._position[:] = position[:]
        self._velocity[:] = velocity[:]

    @property
    def ndim(self):
        return self._ndim

    @property
    def position(self):
        return self._position

    @property
    def velocity(self):
        return self._velocity

    @property
    def speed(self):
        return np.linalg.norm(self.velocity)

    @property
    def direction(self):
        return self.velocity / self.speed

    def distance(self, other):
        """Distance from the other objects."""
        if isinstance(other, Agent):
            return np.linalg.norm(self.position - other.position)
        # If other is not agent, let other tell the distance.
        return other.distance(self.position)

    def _regularize(self):
        if self.max_speed:
            if self.speed > self.max_speed:
                self._velocity = self._velocity / self.speed * self.max_speed

        if self.max_acceleration:
            acceleration = np.linalg.norm(self._acceleration)
            if acceleration > self.max_acceleration:
                self._acceleration = self._acceleration / acceleration * self.max_acceleration

    def move(self, dt):
        self._velocity += self._acceleration * dt
        # Velocity cap
        self._regularize()

        self._position += self._velocity * dt

    def can_see(self, other):
        """Whether the boid can see the other."""
        return self.distance(other) < self.vision

    def observe(self, environment):
        """Observe the population and take note of neighbors."""
        self.neighbors = [other for other in environment.population
                          if self.can_see(other) and id(other) != id(self)]
        # To simplify computation, it is assumed that agent is aware of all
        # obstacles including the boundaries. In reality, the agent is only
        # able to see the obstacle when it is in visual range. This doesn't
        # affect agent's behavior, as agent only reacts to obstacles when in
        # proximity, and no early planning by the agent is made.
        self.obstacles = [obstacle for obstacle in environment.obstacles
                          if self.can_see(obstacle)]    