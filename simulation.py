import os
import argparse
import json
import time

import numpy as np

from classes import *


def random_obstacle(position1, position2, r):
    """Return an obstacle of radius r randomly placed between position1 and position2"""
    d = position1 - position2
    d_len = np.sqrt(d.dot(d))
    cos = d[0] / d_len
    sin = d[1] / d_len

    # Generat random x and y assuming d is aligned with x axis.
    x = np.random.uniform(2+r, d_len-r)
    y = np.random.uniform(-r, r)

    # Rotate the alignment back to the actural d.
    true_x = x * cos + y * sin + position2[0]
    true_y = x * sin - y * cos + position2[1]

    return Sphere(r, [true_x, true_y], ndim=2)


def main():
    if ARGS.model not in ('boid', 'vicsek'):
        raise argparse.ArgumentTypeError('model must be one of ("boid", "viscek")')

    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    with open(ARGS.config) as f:
        model_config = json.load(f)

    if ARGS.model == 'boid':
        Boid.set_model(**model_config)
        Model = Boid
    elif ARGS.model == 'vicsek':
        Model = Vicsek

    region = (-100, 100, -100, 100)

    position_data_all = []
    velocity_data_all = []

    prev_time = time.time()
    for i in range(ARGS.instances):
        if i % 100 == 0:
            print('Simulation {}/{}... {:.1f}s'.format(i,
                                                       ARGS.instances, time.time()-prev_time))
            prev_time = time.time()

        env = Environment2D(region)
        for _ in range(ARGS.agents):
            agent = Model(ndim=2, vision=ARGS.vision, size=ARGS.size,
                        max_speed=10, max_acceleration=20)
            agent.initialize(np.random.uniform(-80, 80, 2),
                            np.random.uniform(-15, 15, 2))
            env.add_agent(agent)

        goal = Goal(np.random.uniform(-40, 40, 2), ndim=2)
        env.add_goal(goal)
        # Create a sphere obstacle near segment between avg boids position and goal position.
        avg_boids_position = np.mean(np.vstack([agent.position for agent in env.population]), axis=0)

        spheres = []
        for _ in range(ARGS.obstacles):
            sphere = random_obstacle(avg_boids_position, goal.position, 8)
            spheres.append(sphere)
            env.add_obstacle(sphere)

        position_data = []
        velocity_data = []
        for _ in range(ARGS.steps):
            env.update(ARGS.dt)
            position_data.append([goal.position for goal in env.goals] +
                                 [sphere.position for sphere in spheres] +
                                 [agent.position.copy() for agent in env.population])
            velocity_data.append([np.zeros(2) for goal in env.goals] +
                                 [np.zeros(2) for sphere in spheres] +
                                 [agent.velocity.copy() for agent in env.population])

        position_data_all.append(position_data)
        velocity_data_all.append(velocity_data)

    print('Simulations {0}/{0} completed.'.format(ARGS.instances))

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_position.npy'), position_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_velocity.npy'), velocity_data_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='boid',
                        help='agent model')
    parser.add_argument('--agents', type=int, default=10,
                        help='number of agents')
    parser.add_argument('--obstacles', type=int, default=0,
                        help='number of obstacles')
    parser.add_argument('--vision', type=float, default=None,
                        help='vision range to determine frequency of interaction')
    parser.add_argument('--size', type=float, default=3,
                        help='agent size zone size')
    parser.add_argument('--steps', type=int, default=200,
                        help='number of simulation steps')
    parser.add_argument('--instances', type=int, default=1,
                        help='number of simulation instances')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='time resolution')
    parser.add_argument('--config', type=str, default='config/default_boid.json',
                        help='path to config file')
    parser.add_argument('--save-dir', type=str,
                        help='name of the save directory')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for save files')

    ARGS = parser.parse_args()

    main()
