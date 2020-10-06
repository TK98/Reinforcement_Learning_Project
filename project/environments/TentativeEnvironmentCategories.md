# Possible classifications of enviroments we can use

* **Time-Based Rewards**: we get a costant reward per every time step
    - **Positive rewards**: try to keep the episode running for as much time as possible
        - *pole-cart;*
    - **Positive rewards**: try to finish the episode ASAP
        - *windy, mazes, mountain-car, Acrobot;*

* **Binary-choice with delayed reward**: Choose at the initial state left or right, get a delayed reward after some steps in that direction
    - *1-d grid world, A-split (albeit this one has no actions!)*

* **Counter-intuitive environments**: Enviroments where you have to do weird stuff, like going backwards in order to go forward
    - *Mountain-Car*  

* **Delay of reward**
    - **Immediate**: you act, you get
        - *Bipedal walker*
    - **No immediate reward**
        - *Any time-based reward, any binary-choice w/ delayed reward*

# Other distinctions we could make:
* Hard vs simple environments (e.g. 2DBipedal vs 2DBidepal-Hard)
* Discrete state vs Continuos state
* atari games: very complex, might be good to see the "practicality", however not sure whether the linear approx works here

# Noteworthy
* Car track is a mix between Time-based and Immediate reward.
* LunarLander: saves "resources" (there is a continous version)
* I would say that anything beyond ATARI is too hard for us... Maybe we can try 1 MuJoCo env.

# Possible enviroments we could use
I would say all the toy examples (we coded)
Then pole-cart, mountain-car (both discrete and continous possibly, to draw a comparison)
Bipedal walker (maybe with a hard one? if we have time)
1 Atari game if we can