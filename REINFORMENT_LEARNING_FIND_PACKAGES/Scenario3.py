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

        self.num_visited = None
        self.QMatrix = None
        self.fourRoomObject = four_room
        self.RewardMatrix = None
        self.packet_pos = np.zeros((3, 1))
        self.size_environment = 12
        self.width = 4
        self.height = 144
        self.number_packages = 3
        self.moves = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])
        self.episodes = 10000
        self.epsilon = 1
        self.start_epsilon_decay = 1
        self.end_epsilon_decay = self.episodes // 2
        self.epsilon_decay_value = self.epsilon / (self.end_epsilon_decay - self.start_epsilon_decay)
        self.show = 20
        self.prev_gridCell = []
        self.prev_current_num_package = []

    def state_actions(self, number_packages: int, state: int) -> [int]:
        """
        - Get actions from state
        - Action of state are columns of RewardsMatrix
        - Only zero values are valid move
        :param number_packages:
        :param state: current state for action
        :return: integer array of actions
        """
        actions = []
        index = 0
        for element in self.RewardMatrix[number_packages, state, :]:
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

    def reward(self) -> np.array([[[int]]]):
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
        Reward_Matrix = np.zeros((self.number_packages, self.height, self.width), dtype=int)
        for package in range(self.number_packages):
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
                        Reward_Matrix[package, state, index] = -1
                    index += 1
        return Reward_Matrix

    def reward_action(self, current_state: int, next_state: int, action: int, number_package: int,
                      grid_cell: int, _isTerminal: bool) -> float:
        """
        - Get rewards for action take from state to next state
        - If current state equals next state , it means we bounced on the walls
        - then we reward that action by negative one
        -Else reward by grid cell value which zero for moves to state and 1 for move terminal state
        :param _isTerminal: boolean tells weather we at terminal state
        :param number_package:number of package left
        :param current_state:  state where action is taken
        :param next_state:  destination state after taking action
        :param action:  action take to move from state(current state) to next state
        :param grid_cell: is the rewards for valid moves
        """
        if current_state == next_state:
            self.RewardMatrix[number_package, current_state, action] = -1
            return 0.0
        else:
            return grid_cell

    def QExploration(self, epoches: int, gamma: float = 0.9) -> None:

        """
        - We learn the environment until agent get package
        - Choose random number and if random is greater epsilon , it means we have learned at least so
         exploit information
        - Else we continue to learn the environment
        - After each episode or epoch, epsilon is adjusted by epsilon decay
        :param epoches: number of the episodes
        :param gamma: is the gamma - decides how much agent care about future rewards
        """

        Q_table = self.QMatrix.copy()
        Visit_table = self.num_visited.copy()
        X, Y = self.fourRoomObject.getPosition()

        state = Y * self.size_environment + X
        goal_state_isReached = False
        current_num_packages = self.fourRoomObject.getPackagesRemaining()
        self.number_packages = 0 if current_num_packages == 3 else 1 if current_num_packages == 2 else 2
        collected_packages = None
        next_current_num_packages = 0

        while not goal_state_isReached:

            state_action = self.state_actions(self.number_packages, state)

            if random.random() > self.epsilon:
                action = Q_table[self.number_packages, state, :].argmax(0)
            else:
                action = state_action[np.random.randint(0, len(state_action))]

            gridCell, current_pos, current_num_packages, is_terminal = self.fourRoomObject.takeAction(action)
            next_state = current_pos[1] * self.size_environment + current_pos[0]

            if current_num_packages == 2:
                next_current_num_packages = 0
            elif current_num_packages == 1:
                next_current_num_packages = 1
            elif current_num_packages == 0:
                next_current_num_packages = 2

            if current_num_packages == 2 and gridCell > 0 and gridCell != 1:
                break
            if current_num_packages == 1 and gridCell > 0 and gridCell != 2:
                break
            reward = self.reward_action(state, next_state, action, self.number_packages, gridCell, is_terminal)

            Visit_table[self.number_packages, state, action] += 1

            learning_rate = 1 / Visit_table[self.number_packages, state, action]

            Q_table[self.number_packages, state, action] += learning_rate * (
                    reward + gamma * (
                    Q_table[self.number_packages, next_state, :].max(0) - Q_table[self.number_packages, state, action]
            ))
            if current_num_packages == 2 and collected_packages is None:
                Q_table, Visit_table, collected_packages, self.number_packages = \
                    self.store_move_info(Q_table, Visit_table, gridCell, current_num_packages, 1,
                                         next_current_num_packages)

            if current_num_packages == 1 and collected_packages == 1:
                Q_table, Visit_table, collected_packages, self.number_packages = \
                    self.store_move_info(Q_table, Visit_table, gridCell, current_num_packages, 2,
                                         next_current_num_packages)

            if current_num_packages == 0 and collected_packages == 2:
                Q_table, Visit_table, collected_packages, self.number_packages = \
                    self.store_move_info(Q_table, Visit_table, gridCell, current_num_packages, 3,
                                         next_current_num_packages)

            goal_state_isReached = is_terminal
            state = next_state

        if self.end_epsilon_decay >= epoches >= self.start_epsilon_decay:
            self.epsilon -= self.epsilon_decay_value

    def store_move_info(self, Q_table, visit_table, gridCell, current_num_packages, collected_packages,
                        next_packages) -> \
            (np.array([[[float]]]), np.array([[[float]]]), int, int):
        """
        - Update self.QMatrix and self.num_visited
        - Store gridCell and current number packages
        - Update temporal Q_table and visit_table  using  self.QMatrix and self.num_visited
        :param Q_table:  temporal Q_table for each epoch
        :param visit_table: temporal visit_table for each epoch
        :param gridCell: current gridCell ( package value( 1 or 2 or 3))
        :param current_num_packages: number packages  left for collection
        :param collected_packages:  number packages collected
        :param next_packages: next package to collect
        :return: Q_table, visit_table, collected_packages, next_packages
        """

        self.QTable_updater(Q_table, visit_table)
        self.prev_gridCell.append(gridCell)
        self.prev_current_num_package.append(current_num_packages)
        Q_table = self.QMatrix.copy()
        Visit_table = self.num_visited.copy()
        next_packages = next_packages + 1
        return Q_table, Visit_table, collected_packages, next_packages

    def QTable_updater(self, Q_table, visit_table) -> None:
        """
        -Update self.QMatrix and self.num_visited using data from temporal  Q_table and Visit_table respectively
        :param Q_table: temporal Q_table for each epoch
        :param visit_table: temporal visit_table for each epoch
        :return: None
        """
        self.QMatrix = Q_table.copy()
        self.num_visited = visit_table.copy()

    def QExploit(self) -> None:
        """
        - Exploit the know information to find the package
        - Loop and take maximum action as long the package is not found
        - When package is found , we break
        """
        X, Y = self.fourRoomObject.getPosition()
        goal_state_isReached = False
        state = self.size_environment * Y + X
        current_num_packages = self.fourRoomObject.getPackagesRemaining()
        self.number_packages = 0 if current_num_packages == 3 else 1 if current_num_packages == 2 else 2
        circle = []
        while not goal_state_isReached:
            circle.append(state)
            action = self.QMatrix[self.number_packages, state, :].argmax(0)

            reward, current_pos, current_num_packages, is_terminal = self.fourRoomObject.takeAction(action)

            goal_state_isReached = is_terminal
            if goal_state_isReached:
                break

            next_number_packages = 0 if current_num_packages == 3 else 1 if current_num_packages == 2 else 2

            next_state = current_pos[1] * self.size_environment + current_pos[0]

            state = next_state

            self.number_packages = next_number_packages

    def Q_learning(self) -> np.array([int]):
        """
        - Initialise QMatrix to zeros with size of 144 by 4  ( height X width)\
        - Initialise number_visited - array of number of visit to each state
        - Initialise the RewardMatrix
        - Run maximum episodes to allow agent to learn the environment
        :return: integer array of number visit to each state
        """

        self.QMatrix = np.zeros((self.number_packages, self.height, self.width), dtype=float)
        self.num_visited = np.zeros((self.number_packages, self.height, self.width), dtype=int)
        self.RewardMatrix = self.reward()
        bes_epoch = None
        early_stop = 144 * 4
        best_learning = None

        for epoch in range(1, self.episodes):
            self.QExploration(epoch)
            average_states_visit = np.mean(self.QMatrix)
            if best_learning is None or average_states_visit > best_learning:
                bes_epoch = epoch
                best_learning = average_states_visit
            if bes_epoch + early_stop <= epoch:
                break
            self.fourRoomObject.newEpoch()

        return self.num_visited

    def evaluate_agent(self) -> None:
        """
        - Test our agent by Exploiting the known information
        """
        self.fourRoomObject.newEpoch()
        self.QExploit()


def main():
    # initialise fourRoom
    fourRoomsObj = FourRooms("multi")
    # initialise findPackage
    FindPackageObject = FindPackage(fourRoomsObj)
    # Agent learn
    print("Started training ...")
    FindPackageObject.Q_learning()

    # evaluate agent
    FindPackageObject.evaluate_agent()
    # Show Path

    fourRoomsObj.showPath(-1, "./Images/scenario3.png")
    print("Done !!!")
    print("Picture saved, check ./Images/scenario3.png")


if __name__ == "__main__":
    main()

