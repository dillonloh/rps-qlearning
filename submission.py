import numpy as np
import random
qtable = np.zeros(shape=[3, 5])
my_last_move = 0
last_state = 0
last_action = 0 # corresponds to random action
def my_agent(observation, configuration):
    
    def reward_func(my_last_move, opp_last_move):
        """reward function for win/draw/lose"""
        result_table = np.array([['Draw', 'Lose', "Win"],
                                 ['Win', 'Draw', 'Lose'],
                                 ['Lose', 'Win', 'Draw']])

        result = result_table[my_last_move][opp_last_move]
        # returns [reward, row in q-table]
#         print(result)
        if result == 'Draw':
            return [0, 0]
        elif result == 'Win':
            return [1, 1]
        elif result == 'Lose':
            return [-1, 2]        
        
    def action_choices(action, opp_last_move, my_last_move):
        
        if action == 0:
            return move_random()
        elif action == 1:
            return move_react(opp_last_move)
        elif action == 2:
            return move_counterreact(my_last_move)
        elif action == 3:
            return move_same(my_last_move)
        elif action == 4:
            return move_copy(opp_last_move)
    
    def move_random():
        return random.randint(0, 2)
    
    def move_react(opp_last_move):
        if opp_last_move == 2:
            return 0
        else:
            return opp_last_move + 1
    
    def move_counterreact(my_last_move):
        if my_last_move == 0:
            return 2
        else:
            return my_last_move - 1
    
    def move_same(my_last_move):
        return my_last_move
    
    def move_copy(opp_last_move):
        return opp_last_move
    
    
    
    global qtable
    global my_last_move, last_action, last_state
    
    
    if observation.step <= 100:
        EPSILON = 0.8
    else:
        EPSILON = 0.2
        
    LEARNING_RATE = 0.1
    
    if observation.step == 0:
        return 0
    
    opp_last_move = observation['lastOpponentAction']
    reward, new_state = reward_func(my_last_move, opp_last_move)
    qtable[last_state][last_action] = qtable[last_state][last_action] + LEARNING_RATE*(reward) 

    # choose a new action from qtable

    if random.random() <= EPSILON:
        new_action = random.randint(0, 4)
    else:
        new_action = np.argmax(qtable[new_state],)
        
    new_move = action_choices(new_action, opp_last_move, my_last_move)
    
    my_last_move = new_move
    last_action = new_action
    last_state = new_state # reassign the current state to be the next iteration's "last_state"     
    
    return new_move  
