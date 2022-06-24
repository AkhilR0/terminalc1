import gamelib
import random
import math
import warnings
from sys import maxsize
import json
from long_term import *
import numpy as np
import torch
from justcoordinates import correct_coordinates


"""



> State <
enemy health
 - game_state.enemy_health
enemy structure bits
 - game_state.get_resource(0,1)
enemy mobile bits
 - game_state.get_resource(1,1)
your health
 - game_state.my_health
your structure bits
 - game_state.get_resource(0,0)
your mobile bits
 - game_state.get_resource(1,0)
enemy locations (what locations are occupied and what unit is in each location)
 - game_state.game_map[x, y]
your locations (what locations are occupied and what unit is in each location)
 - game_state.game_map[x, y]


, game_state.game_map[13, 27], game_state.game_map[14, 27], game_state.ga

, 0,1,0, 


State array -> [CurrentHealth, EnemyHealth, EnemyStructure, EnemyMobile , Your Structurebits , Mobilebits , locations]

for the locations, how will the algo know whether something is about to be removed or not

state = [game_state.my_health, game_state.enemy_health, game_state.get_resource(0,1), game_state.get_resource(1,1), game_state.get_resource(0,0), game_state.get_resource(1,0), game_state.game_map[x, y]]

health range - 0 to 40. (or 1 to 40)

27x27)/2

[]

[wall, turret, support]

turret

0,1,0



1 -> 200
0 -> 1

normalizeddata = (my health - min)/ (max - min)
normalizedata *= 64

int(normalizeddata) -> bin

00000 -> 0
11111 -> 63

[0,1,0,1,0,0,0,0,0,1,1,1,0,]



print(format(num, '06b'))

# normalizeddata = 17 
# normalizeddata = bin(normalizeddata)
# normalizeddata= normalizeddata.replace("0b", "")

def normalize(val, min, max):
	return (val - min)/(max - min)

normalVal = normalize(12, 1, 30)

bin64 = int(normalVal*64)
if bin64 > 63:
	bin64 = 63

print(format(bin64, "06b"))


import math

x = 17 
x = bin(x)
x= x.replace("0b", "")

[]

should you add turn number into this?

also you have to implement whether the unit is upgraded or not. so not 6 units but 12 units. 
also whether the unit is shielded or not
also the amount of health the unit currently has
game_state.game_map[x, y] returns a LIST of units. what does that mean. how can you have multiple units in one place?.
so how do the mobile units work? when you check for their location, are they moving or on the edge?

Locations is more complicated. There are six possible units that each location could have. so for each location, 
represent it with a 6-unit array. And there will be (27x27)/2 of these unit arrays because that's how many locations 
there are on the diamond-shaped grid. The order will go from top to bottom, left to right.

> Actions <
six diffrerent types of units, can place any of them on all the availabe free locations that are not occupied 
 - game_state.attempt_spawn
upgrade any existing units 
 - game_state.attempt_upgrade
remove a unit
 - game_state.attempt_remove

how do u implement what the agent can't do? like how does it know whether there's already a unit in a place it's trying to spawn something in? do u give a negative reward for that?

{(13,27): [1,1,1,1,1],(13,27): [1,1,1,1,1], (13,27): [1,1,1,1,1], (13,27): [1,1,1,1,1], (13,27): [1,1,1,1,1],  }

go all the way until 128 not 64

[[1,1,1,1,1,1,1,1]] for the edges
[[1,1,1,1,1]] for everything else

6 times (27x27)/4


Action array ->
action  = [game_state.attempt_spawn(spawn_pos), game_state.attempt_upgrade(upgrade_pos), game_state.attempt_remove(remove_pos)]

For each location (again, (27x27)/2 of them) there will be a 8-unit array. the first six are for spawning, and each one represnting each of the 6 units you could spawn. 
the seventh is whether the agent wants to upgrade the unit that is in that specific location. the eighth is whether the agent wants to remove the unit that is in that specific location.

> Other items that can be accessed <
the current turn number
 - game_state.turn_number


> Reward <
 + Winning the game (+40)
 + taking health off of the oppponent (+1)
 + surviving (+1/42)
 + how quickly you win the game
 - losing the game (-40) maybe make this number a lot smaller?
 - losing health (-1)

add functionality so agent can reset the game over and over so it can keep training itself <-- my loop idea -sub

> Training Options <
 - Train against itself or another different type of RL model
 - Train against a random model
 - Train against the in-built training bot (is this good enough?)

Perform random moves towards the beginning, then use model.predict(). make sure to perfect the trade-off (epsilon value)

store all values in an array as an input for both state and receive an array for the action, each number in the array representing something different which would need to be translated for the game

"""


"""
Most of the algo code you write will be in this file unless you create new
modules yourself. Start by modifying the 'on_turn' function.

Advanced strategy tips: 

  - You can analyze action frames by modifying on_action_frame function

  - The GameState.map object can be manually manipulated to create hypothetical 
  board states. Though, we recommended making a copy of the map to preserve 
  the actual current map state.
"""


class AlgoStrategy(gamelib.AlgoCore):
    def __init__(self):
        super().__init__()
        seed = random.randrange(maxsize)
        random.seed(seed)
        gamelib.debug_write('Random seed: {}'.format(seed))

    def on_game_start(self, config):
        """ 
        Read in config and perform any initial setup here 
        """
        gamelib.debug_write('Configuring your custom algo strategy...')
        self.config = config
        global WALL, SUPPORT, TURRET, SCOUT, DEMOLISHER, INTERCEPTOR, MP, SP
        WALL = config["unitInformation"][0]["shorthand"]
        SUPPORT = config["unitInformation"][1]["shorthand"]
        TURRET = config["unitInformation"][2]["shorthand"]
        SCOUT = config["unitInformation"][3]["shorthand"]
        DEMOLISHER = config["unitInformation"][4]["shorthand"]
        INTERCEPTOR = config["unitInformation"][5]["shorthand"]
        MP = 1
        SP = 0
        # This is a good place to do initial setup
        self.scored_on_locations = []

        # reset reward to 0

    def on_turn(self, turn_state):
        """
        This function is called every turn with the game state wrapper as
        an argument. The wrapper stores the state of the arena and has methods
        for querying its state, allocating your current resources as planned
        unit deployments, and transmitting your intended deployments to the
        game engine.
        """
        game_state = gamelib.GameState(self.config, turn_state)
        gamelib.debug_write('Performing turn {} of your custom algo strategy'.format(
            game_state.turn_number))
        # game_state.suppress_warnings(True)  #Comment or remove this line to enable warnings.

        # Training
        #

        # translate_move(agent.get_action(state_old))
        # gamelib.debug_write(game_state.UNIT_TYPE_TO_INDEX)

        dictionary_for_units = {WALL: [1,0,0], SUPPORT: [0,1,0], TURRET: [0,0,1]}
        input_array = []


        #code for adding healtha and stuff
        state = [['game_state.my_health', 0, 40], ['game_state.enemy_health', 0, 40], ['game_state.get_resource(0,1)', 0, 40], ['game_state.get_resource(1,1)', 0, 40], ['game_state.get_resource(0,0)', 0, 40], ['game_state.get_resource(1,0)', 0, 40], ['game_state.game_map[x, y]', 0, 40]]
       
        for i in state:
            temp_arr = self.normalize_properly(i[0], i[1], i)
            for i in temp_arr:
                input_array.append(i)
                
        

        for i in all_coordinates:
            x = i[0]
            y = i[1]
            current_units = game_state.game_map[x,y]
            current_arr = dictionary_for_units(current_units[0])
            for haha in current_arr:
                input_array.append(haha)
       

        totalStates += input_array



        enemyHealths[game_state.turn_number] = game_state.enemy_health
        agent_reward += 1  # For surviving
        if (game_state.turn_number > 0):
            agent_reward += enemyHealths[game_state.turn_number] - \
                enemyHealths[game_state.turn_number - 1]  # for getting health off of your opponent
            agent_reward -= yourHealths[game_state.turn_number] - \
                yourHealths[game_state.turn_number-1]  # for losing health

        final_move = agent.get_action(state_old)

       

        number_of_edges = 20 * 8  # change this
        number_of_normal = 50 * 5  # change this



        edge_moves = np.asarray(final_move)
        edge_moves = edge_moves[:number_of_edges]
        edge_moves.resize(number_of_edges/8, 8)

        normal_moves = np.asarray(final_move)
        normal_moves = normal_moves[number_of_edges:]
        edge_moves.resize(number_of_normal/5, 5)

        for i in range(0, number_of_edges/8):
            current_array = np.asarray(edge_moves[i])
            if (torch.max(current_array) > 1):
                if (torch.argmax(current_array) == 0):
                    game_state.attempt_spawn(WALL, edge_locations[i])  # wall
                elif (torch.argmax(current_array) == 1):
                    game_state.attempt_spawn(
                        SUPPORT, edge_locations[i])  # support
                elif (torch.argmax(current_array) == 2):
                    game_state.attempt_spawn(
                        TURRET, edge_locations[i])  # turret
                elif (torch.argmax(current_array) == 3):
                    game_state.attempt_spawn(SCOUT, edge_locations[i])  # scout
                elif (torch.argmax(current_array) == 4):
                    game_state.attempt_spawn(
                        DEMOLISHER, edge_locations[i])  # demolisher
                elif (torch.argmax(current_array) == 5):
                    game_state.attempt_spawn(
                        INTERCEPTOR, edge_locations[i])  # interceptor
                elif (torch.argmax(current_array) == 6):
                    game_state.attempt_upgrade(edge_locations[i])  # upgrage
                else:
                    game_state.attempt_remove(edge_locations[i])

        for i in range(0, number_of_normal/5):
            current_array = np.asarray(normal_moves[i])
            if (torch.max(current_array) > 1):
                if (torch.argmax(current_array) == 0):
                    game_state.attempt_spawn(WALL, normal_locations[i])  # wall
                elif (torch.argmax(current_array) == 1):
                    game_state.attempt_spawn(
                        SUPPORT, normal_locations[i])  # support
                elif (torch.argmax(current_array) == 2):
                    game_state.attempt_spawn(
                        TURRET, normal_locations[i])  # turret
                elif (torch.argmax(current_array) == 3):
                    game_state.attempt_upgrade(normal_locations[i])  # upgrade
                else:
                    game_state.attempt_remove(normal_locations[i])  # remove

        # final_move should be an array of 1s and 0s

        # change this. how does reward work
        reward, done, score = game.play_step(final_move)

        self.starter_strategy(game_state)

        game_state.submit_turn()

    """
    NOTE: All the methods after this point are part of the sample starter-algo
    strategy and can safely be replaced for your custom algo.
    """


# def translate_move(self, ):
    def normalize_properly(self, value, mini, maxi):    
        normalVal = (self.value - self.mini)/(self.maxi - self.mini)   
        bin128 = int(normalVal*128)

        if bin128 > 127:
            bin128 = 127

        bin128 = int(bin(bin128))
        bin128 = bin128.replace("0b", "")
        
        return bin128




    def starter_strategy(self, game_state):
        """
        For defense we will use a spread out layout and some interceptors early on.
        We will place turrets near locations the opponent managed to score on.
        For offense we will use long range demolishers if they place stationary units near the enemy's front.
        If there are no stationary units to attack in the front, we will send Scouts to try and score quickly.
        """
        # First, place basic defenses
        self.build_defences(game_state)
        # Now build reactive defenses based on where the enemy scored
        self.build_reactive_defense(game_state)

        # If the turn is less than 5, stall with interceptors and wait to see enemy's base
        if game_state.turn_number < 5:
            self.stall_with_interceptors(game_state)
        else:
            # Now let's analyze the enemy base to see where their defenses are concentrated.
            # If they have many units in the front we can build a line for our demolishers to attack them at long range.
            if self.detect_enemy_unit(game_state, unit_type=None, valid_x=None, valid_y=[14, 15]) > 10:
                self.demolisher_line_strategy(game_state)
            else:
                # They don't have many units in the front so lets figure out their least defended area and send Scouts there.

                # Only spawn Scouts every other turn
                # Sending more at once is better since attacks can only hit a single scout at a time
                if game_state.turn_number % 2 == 1:
                    # To simplify we will just check sending them from back left and right
                    scout_spawn_location_options = [[13, 0], [14, 0]]
                    best_location = self.least_damage_spawn_location(
                        game_state, scout_spawn_location_options)
                    game_state.attempt_spawn(SCOUT, best_location, 1000)

                # Lastly, if we have spare SP, let's build some supports
                support_locations = [[13, 2], [14, 2], [13, 3], [14, 3]]
                game_state.attempt_spawn(SUPPORT, support_locations)

    def build_defences(self, game_state):
        """
        Build basic defenses using hardcoded locations.
        Remember to defend corners and avoid placing units in the front where enemy demolishers can attack them.
        """
        # Useful tool for setting up your base locations: https://www.kevinbai.design/terminal-map-maker
        # More community tools available at: https://terminal.c1games.com/rules#Download

        # Place turrets that attack enemy units
        turret_locations = [[0, 13], [27, 13], [
            8, 11], [19, 11], [13, 11], [14, 11]]
        # attempt_spawn will try to spawn units if we have resources, and will check if a blocking unit is already there
        game_state.attempt_spawn(TURRET, turret_locations)

        # Place walls in front of turrets to soak up damage for them
        wall_locations = [[8, 12], [19, 12]]
        game_state.attempt_spawn(WALL, wall_locations)
        # upgrade walls so they soak more damage
        game_state.attempt_upgrade(wall_locations)

    def build_reactive_defense(self, game_state):
        """
        This function builds reactive defenses based on where the enemy scored on us from.
        We can track where the opponent scored by looking at events in action frames 
        as shown in the on_action_frame function
        """
        for location in self.scored_on_locations:
            # Build turret one space above so that it doesn't block our own edge spawn locations
            build_location = [location[0], location[1]+1]
            game_state.attempt_spawn(TURRET, build_location)

    def stall_with_interceptors(self, game_state):
        """
        Send out interceptors at random locations to defend our base from enemy moving units.
        """
        # We can spawn moving units on our edges so a list of all our edge locations
        friendly_edges = game_state.game_map.get_edge_locations(
            game_state.game_map.BOTTOM_LEFT) + game_state.game_map.get_edge_locations(game_state.game_map.BOTTOM_RIGHT)

        # Remove locations that are blocked by our own structures
        # since we can't deploy units there.
        deploy_locations = self.filter_blocked_locations(
            friendly_edges, game_state)

        # While we have remaining MP to spend lets send out interceptors randomly.
        while game_state.get_resource(MP) >= game_state.type_cost(INTERCEPTOR)[MP] and len(deploy_locations) > 0:
            # Choose a random deploy location.
            deploy_index = random.randint(0, len(deploy_locations) - 1)
            deploy_location = deploy_locations[deploy_index]

            game_state.attempt_spawn(INTERCEPTOR, deploy_location)
            """
            We don't have to remove the location since multiple mobile 
            units can occupy the same space.
            """

    def demolisher_line_strategy(self, game_state):
        """
        Build a line of the cheapest stationary unit so our demolisher can attack from long range.
        """
        # First let's figure out the cheapest unit
        # We could just check the game rules, but this demonstrates how to use the GameUnit class
        stationary_units = [WALL, TURRET, SUPPORT]
        cheapest_unit = WALL
        for unit in stationary_units:
            unit_class = gamelib.GameUnit(unit, game_state.config)
            if unit_class.cost[game_state.MP] < gamelib.GameUnit(cheapest_unit, game_state.config).cost[game_state.MP]:
                cheapest_unit = unit

        # Now let's build out a line of stationary units. This will prevent our demolisher from running into the enemy base.
        # Instead they will stay at the perfect distance to attack the front two rows of the enemy base.
        for x in range(27, 5, -1):
            game_state.attempt_spawn(cheapest_unit, [x, 11])

        # Now spawn demolishers next to the line
        # By asking attempt_spawn to spawn 1000 units, it will essentially spawn as many as we have resources for
        game_state.attempt_spawn(DEMOLISHER, [24, 10], 1000)

    def least_damage_spawn_location(self, game_state, location_options):
        """
        This function will help us guess which location is the safest to spawn moving units from.
        It gets the path the unit will take then checks locations on that path to 
        estimate the path's damage risk.
        """
        damages = []
        # Get the damage estimate each path will take
        for location in location_options:
            path = game_state.find_path_to_edge(location)
            damage = 0
            for path_location in path:
                # Get number of enemy turrets that can attack each location and multiply by turret damage
                damage += len(game_state.get_attackers(path_location, 0)) * \
                    gamelib.GameUnit(TURRET, game_state.config).damage_i
            damages.append(damage)

        # Now just return the location that takes the least damage
        return location_options[damages.index(min(damages))]


# attempt_spawn
# game_state.contains_stationary_unit


    def detect_enemy_unit(self, game_state, unit_type=None, valid_x=None, valid_y=None):
        total_units = 0
        for location in game_state.game_map:
            if game_state.contains_stationary_unit(location):
                for unit in game_state.game_map[location]:
                    if unit.player_index == 1 and (unit_type is None or unit.unit_type == unit_type) and (valid_x is None or location[0] in valid_x) and (valid_y is None or location[1] in valid_y):
                        total_units += 1
        return total_units

    def filter_blocked_locations(self, locations, game_state):
        filtered = []
        for location in locations:
            if not game_state.contains_stationary_unit(location):
                filtered.append(location)
        return filtered

    def on_action_frame(self, turn_string):
        """
        This is the action frame of the game. This function could be called 
        hundreds of times per turn and could slow the algo down so avoid putting slow code here.
        Processing the action frames is complicated so we only suggest it if you have time and experience.
        Full doc on format of a game frame at in json-docs.html in the root of the Starterkit.
        """
        # Let's record at what position we get scored on
        state = json.loads(turn_string)
        events = state["events"]
        breaches = events["breach"]
        for breach in breaches:
            location = breach[0]
            unit_owner_self = True if breach[4] == 1 else False
            # When parsing the frame data directly,
            # 1 is integer for yourself, 2 is opponent (StarterKit code uses 0, 1 as player_index instead)
            if not unit_owner_self:
                gamelib.debug_write("Got scored on at: {}".format(location))
                self.scored_on_locations.append(location)
                gamelib.debug_write(
                    "All locations: {}".format(self.scored_on_locations))


if __name__ == "__main__":
    algo = AlgoStrategy()
    algo.start()
