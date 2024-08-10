from packages.action import step
from packages.ScreenGrab import getScreen
from packages.qnn import Agent

if __name__ == '__main__':

    agent = Agent(gamma=0.99,epsilon=1.0, lr= 0.003, input_dims= 447200, batch_size = 64 , n_actions = 8)
    
    learning_steps = 10000
    score = 0
    observation = getScreen()
    for i in range(learning_steps):
        action = agent.choose_action(observation)
        observation_ , reward = step(action)
        score += reward
        agent.store_transition(observation, action, reward, observation_)
        agent.learn()
        observation = observation_
        print(score)
