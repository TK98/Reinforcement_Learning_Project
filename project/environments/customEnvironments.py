import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from gym import spaces, logger
import math


# things to do for each env:
#   - step, reset functions, seed
#   - action_space, reward_range
# things I skipped:
#   - render function
#   - observation_space attribute
#   - close function

class OneHotEnv(gym.Env):
    def __init__(self, n_states):
        self.n_states = n_states
        self.seed()

    def _one_hot(self, state):
        onehot = np.zeros(self.n_states)
        onehot[state] = 1
        return onehot

class BairdsCounterExample(OneHotEnv):
    reward_range = (0, 0)

    def __init__(self):
        self.shape = 7
        self.action_space = spaces.Discrete(2)
        super().__init__(7)

    def reset(self):
        self.current = self.np_random.choice(range(0, 6))
        return self._one_hot(self.current)

    def step(self, action):
        if action == 1:
            self.current = 6
        elif action == 0:
            self.current = self.np_random.choice(range(0, 6))

        done = self.current == 6

        return self._one_hot(self.current), 0, done, ""

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class ASplit(OneHotEnv):
    reward_range = (0, 1)

    def __init__(self):
        # A,B,C + Terminal
        self.shape = 4
        self.action_space = spaces.Discrete(1)
        super().__init__(4)

    def reset(self):
        self.current = 0
        return self._one_hot(self.current)

    def step(self, action):
        # there is only one action

        # if state is A
        if self.current == 0:
            # choose randomly between states 1 and 2
            self.current = self.np_random.choice([1, 2])
            done = False
            r = 0
        # if state is B
        elif self.current == 1:
            # go to terminal and give reward
            self.current = 3
            done = True
            r = 1
        # if state is C
        elif self.current == 2:
            # same!
            self.current = 3
            done = True
            r = 0
        elif self.current == 3:
            raise Exception("You can't be in the terminal state. Perhaps you forgot to .reset()?")

        return self._one_hot(self.current), r, done, ""

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# TODO implement this, but with actions...?
class NStateRandomWalk(OneHotEnv):
    reward_range = (0, 1)

    def __init__(self):
        # 0Terminal,A,B,C,D,E,1Terminal
        self.shape = 7
        self.action_space = spaces.Discrete(1)
        super().__init__(7)
        self.leftBound, self.rightBound = 1, 5

    def reset(self):
        # central state!
        self.current = 3
        return self._one_hot(self.current)

    def step(self, action):
        # there is only one action
        direction = self.np_random.choice([-1, 1])
        self.current += direction
        if self.current > self.rightBound:
            done = True
            r = 1
        elif self.current < self.leftBound:
            done = True
            r = 0
        else:
            done = False
            r = 0

        return self._one_hot(self.current), r, done, ""

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class WindyGridWorld(OneHotEnv):
    reward_range = (-1, -1)

    def __init__(self):
        self.shape = 70
        self.action_space = spaces.Discrete(4) # u, d, r, l
        # goal
        self.goal = (7, 3)
        # bounds
        self.x_l, self.x_u = 0, 9
        self.y_l, self.y_u = 0, 6
        # prepare wind
        self.wind = dict()
        for x in range(self.x_l, self.x_u+1):
            if x in [3,4,5,8]:
                self.wind[x] = 1
            elif x in [6,7]:
                self.wind[x] = 2
            else:
                self.wind[x] = 0

    def reset(self):
        self.x, self.y = 0, 3
        self.steps = 0
        return self.get_onehot()

        # return (self.x, self.y)

    def step(self, action):
        # apply wind
        self.y += self.wind[self.x]
        self.y = min(self.y, self.y_u)

        # up
        if action == 0:
            self.y += 1
            self.y = min(self.y, self.y_u)
        # down
        elif action == 1:
            self.y -= 1
            self.y = max(self.y, self.y_l)
        # right
        elif action == 2:
            self.x += 1
            self.x = min(self.x, self.x_u)
        # left
        elif action == 3:
            self.x -= 1
            self.x = max(self.x, self.x_l)
        else:
            raise Exception("There is no such action!")

        self.steps += 1
        # check goal!
        done = (self.x, self.y) == self.goal
        done = done or self.steps > 1000

        return self.get_onehot(), -1, done, ""
        # return (self.x, self.y), -1, done, ""

    def get_onehot(self):
        grid = np.zeros((self.y_u + 1, self.x_u + 1))
        grid[self.y, self.x] = 1

        return grid.ravel()

    def seed(self, seed=None):
        return [seed]

class CartPoleEnv(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        self.shape = 4
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = 'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Max steps threshold
        self.steps_threshold = 200

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array([self.x_threshold * 2,
                         np.finfo(np.float32).max,
                         self.theta_threshold_radians * 2,
                         np.finfo(np.float32).max],
                        dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed()
        self.viewer = None
        self.state = None

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        err_msg = "%r (%s) invalid" % (action, type(action))
        assert self.action_space.contains(action), err_msg

        self.steps += 1

        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
            or self.steps >= self.steps_threshold
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.steps = 0
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width/world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
