# Mountain Car trained using Deep Q-Network

This is mostly a tidied-up code of [pylSER][original]. It solves the
[MoutainCar-v0][mountain] problem of OpenAI gym using Deep Q-Network.

Read more about the solution here: [deep-q][deep].

# How to run

Install [poetry][poetry]. Then run `poetry install`. Due to a [bug][bug] in Tensorflow,
you should also run `pip install tensorflow` to install it separately. Finally run

`python -m mountain-car`

[original]: https://github.com/pylSER/Deep-Reinforcement-learning-Mountain-Car
[poetry]: https://poetry.eustace.io/docs/#installation
[bug]: https://github.com/sdispater/poetry/issues/1330
[mountain]: https://gym.openai.com/envs/MountainCar-v0/
[deep]: https://towardsdatascience.com/qrash-course-deep-q-networks-from-the-ground-up-1bbda41d3677
