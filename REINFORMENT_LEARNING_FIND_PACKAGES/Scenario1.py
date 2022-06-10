"""
Author: Lunga Tsewu
Date: 13 May  2022
"""
import random
from FourRooms import FourRooms as FourRooms
import numpy as np


class FindPackage:

    def __init__(self, four_room: FourRooms):

        """
        -Initialise fields of the class
        :param four_room: fourRoom object
        """
        self.QMatrix = None
        self.fourRoomObject = four_room
        self.episodes = 100
        self.RewardMatrix = None
        self.size_environment = 12
        self.width = 4
        self.height = 144
        self.moves = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        self.epsilon = 1
        self.start_epsilon_decay = 1
        self.end_epsilon_decay = self.episodes // 2
        self.epsilon_decay_value = self.epsilon / (self.end_epsilon_decay - self.start_epsilon_decay)
        self.show = 20

    def state_actions(self, state: int) -> [int]:
        """
        - Get actions from state
        - Action of state are columns of RewardsMatrix
        - Only zero values are valid move
        :param state: current state for action
        :return: integer array of actions
        """
        actions = []
        index = 0
        for element in self.RewardMatrix[state]:
            if element != -1:
                actions.append(index)
            index += 1
        return actions

    def next_state(self, current_state: int, action: int) -> int:
        """
        - get next states when taking action from give state
        - Add the co-ordinate of the state wit action
        :param current_state:  is the state where action is taken
        :param action:  action taken from state
        :return: integer state value
        """
        next_state = np.array(
            [current_state % self.size_environment, current_state // self.size_environment]) + self.moves[action]

        return next_state[1] * self.size_environment + next_state[0]

    def reward(self) -> np.array([[int]]):
        """
            - Add possible move from given state
            - move left columns
           - move right columns
           - move up the rows
           - move down the rows
           - action ( x or y co-ordinates) that are out of bound marked by -1
           - initialise Rewards metrix to zeros
           - Unreachable state are marked by -1 in the Reward Matrix
           :return  integer 2d numpy array of actions for  state
        """
        Reward_Matrix = np.zeros((self.height, self.width), dtype=int)
        for state in range(self.height):
            actions = [
                [state % self.size_environment,
                 state // self.size_environment - 1 if 0 <= state // self.size_environment - 1 < self.size_environment
                 else -1],
                [state % self.size_environment,
                 state // self.size_environment + 1 if 0 <= state // self.size_environment + 1 < self.size_environment
                 else -1],
                [
                    state % self.size_environment - 1 if 0 <= state % self.size_environment - 1 < self.size_environment
                    else -1, state // self.size_environment],
                [
                    state % self.size_environment + 1 if 0 <= state % self.size_environment + 1 < self.size_environment
                    else -1, state // self.size_environment]

            ]
            index = 0
            for action in actions:
                if action[0] < 0 or action[1] < 0:
                    Reward_Matrix[state, index] = -1
                index += 1
        return Reward_Matrix

    def reward_action(self, current_state: int, next_state: int, action: int, grid_cell: int) -> int:
        """
        - Get rewards for action take from state to next state
        - If current state equals next state , it means we bounced on the walls
        - then we reward that action by negative one
        -Else reward by grid cell value which zero for moves to state and 1 for move terminal state
        :param current_state:  state where action is taken
        :param next_state:  destination state after taking action
        :param action:  action take to move from state(current state) to next state
        :param grid_cell: is the rewards for valid moves
        :return: integer reward value
        """
        if current_state == next_state:
            self.RewardMatrix[current_state, action] = -1
            return self.RewardMatrix[current_state, action]
        else:
            self.RewardMatrix[current_state, action] = grid_cell
            return self.RewardMatrix[current_state, action] + 1

    def Q_exploration(self, epoches: int, number_visit: np.array([[int]]), gamma: float = 0.90) -> None:

        """
        - We learn the environment until agent get package
        - Choose random number and if random is greater epsilon , it means we have learned at least so
         exploit information
        - Else we continue to learn the environment
        - After each episode or epoch, epsilon is adjusted by epsilon decay
        :param epoches: number of the episodes
        :param number_visit: number each state has been visited by agent
        :param gamma: is the gamma - decides how much agent care about future rewards
        """

        X, Y = self.fourRoomObject.getPosition()
        state = Y * self.size_environment + X
        goal_state_isReached = False

        while not goal_state_isReached:
            number_visit[state] += 1

            learning_rate = 1 / number_visit[state]

            state_action = self.state_actions(state)

            if random.random() > self.epsilon:
                action = self.QMatrix[state, :].argmax(0)
            else:
                action = state_action[np.random.randint(0, len(state_action))]

            next_state = self.next_state(state, action)

            gridCell, current_pos, current_num_packages, is_terminal = self.fourRoomObject.takeAction(action)

            self.reward_action(state, (current_pos[1] * self.size_environment + current_pos[0]), action, gridCell)

            self.QMatrix[state, action] += learning_rate * (
                    gridCell + gamma * (self.QMatrix[next_state, :].max(0) - self.QMatrix[state, action]))

            goal_state_isReached = is_terminal

            X, Y = current_pos
            state = Y * self.size_environment + X

        if self.end_epsilon_decay >= epoches >= self.start_epsilon_decay:
            self.epsilon -= self.epsilon_decay_value

    def Q_exploit(self) -> None:
        """
        - Exploit the know information to find the package
        - Loop and take maximum action as long the package is not found
        - When package is found , we break
        """
        X, Y = self.fourRoomObject.getPosition()
        goal_state_isReached = False
        optimal_moves = []
        state = self.size_environment * Y + X

        while not goal_state_isReached:

            action = self.QMatrix[state, :].argmax(0)
            reward, current_pos, current_num_packages, is_terminal = self.fourRoomObject.takeAction(action)

            state = current_pos[1] * self.size_environment + current_pos[0]

            goal_state_isReached = is_terminal

            if not goal_state_isReached:
                optimal_moves.append(action)

    def Q_learning(self) -> np.array([int]):
        """
        - Initialise QMatrix to zeros with size of 144 by 4  ( height X width)
        - Initialise number_visited - array of number of visit to each state
        - Initialise the RewardMatrix
        - Run maximum episodes to allow agent to learn the environment
        :return: integer array of number visit to each state
        """
        self.QMatrix = np.zeros((self.height, self.width))
        num_visited = np.zeros(self.height)
        self.RewardMatrix = self.reward()
        bes_epoch = None
        early_stop = 20
        best_learning = None
        for epoch in range(1, self.episodes):
            self.Q_exploration(epoch, num_visited)
            average_states_visit = np.mean(self.QMatrix)

            if best_learning is None or round(average_states_visit, 3) > best_learning:
                best_learning = round(average_states_visit, 3)
                bes_epoch = epoch
            if bes_epoch + early_stop <= epoch:
                break
            self.fourRoomObject.newEpoch()

        return num_visited

    def evaluate_agent(self) -> None:
        """
        - Test our agent by Exploiting the known information
        """
        self.fourRoomObject.newEpoch()
        self.Q_exploit()


def main():
    # initialise fourRoom
    fourRoomsObj = FourRooms("simple")
    # initialise findPackage
    FindPackageObject = FindPackage(fourRoomsObj)
    # Agent learn
    print("Started learning ...")
    FindPackageObject.Q_learning()
    # evaluate agent
    FindPackageObject.evaluate_agent()
    # Show Path
    fourRoomsObj.showPath(-1, "./Images/scenario1.png")
    print("Done !!!")
    print("Picture saved, check ./Images/scenario1.png")


if __name__ == "__main__":
    main()
