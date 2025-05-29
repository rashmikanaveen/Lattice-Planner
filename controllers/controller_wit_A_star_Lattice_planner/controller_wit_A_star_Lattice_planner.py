from controller import Robot, Camera, Accelerometer, DistanceSensor, Motor, PositionSensor
import math
import heapq
from collections import namedtuple

# Constants
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
TILE_SIZE = 0.1

# Map: 0 - free, 1 - obstacle
world = [
    [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]
]
ROWS, COLS = len(world), len(world[0])

class SimpleLatticeState:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = self.normalize_angle(theta)
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def __eq__(self, other):
        return (abs(self.x - other.x) < 0.01 and 
                abs(self.y - other.y) < 0.01 and 
                abs(self.theta - other.theta) < 0.1)
    
    def __hash__(self):
        return hash((round(self.x, 1), round(self.y, 1), round(self.theta, 1)))
    
    def __lt__(self, other):
        """Less than comparison for heapq - compare by position first, then angle"""
        if self.x != other.x:
            return self.x < other.x
        if self.y != other.y:
            return self.y < other.y
        return self.theta < other.theta

class SimpleLattice:
    def __init__(self):
        # Define possible moves: Straight, Left turn, Right turn
        self.moves = {
            'S': {'cost': 1.0, 'theta_change': 0},
            'L': {'cost': math.pi/2, 'theta_change': math.pi/2},
            'R': {'cost': math.pi/2, 'theta_change': -math.pi/2}
        }
        self.node_counter = 0  # To ensure unique heap entries
    
    def get_successors(self, state):
        """Generate successor states from current state"""
        successors = []
        
        for move, params in self.moves.items():
            new_theta = state.theta + params['theta_change']
            
            if move == 'S':
                # Move forward one tile based on current orientation
                if abs(state.theta) < 0.1:  # Facing right (0)
                    new_x, new_y = state.x, state.y + 1
                elif abs(state.theta - math.pi/2) < 0.1:  # Facing up (π/2)
                    new_x, new_y = state.x - 1, state.y
                elif abs(abs(state.theta) - math.pi) < 0.1:  # Facing left (π or -π)
                    new_x, new_y = state.x, state.y - 1
                elif abs(state.theta + math.pi/2) < 0.1:  # Facing down (-π/2)
                    new_x, new_y = state.x + 1, state.y
                else:
                    # Fallback for other angles
                    new_x = state.x - round(math.sin(state.theta))
                    new_y = state.y + round(math.cos(state.theta))
            else:
                # For turns, stay in same position but change orientation
                new_x = state.x
                new_y = state.y
            
            new_state = SimpleLatticeState(new_x, new_y, new_theta)
            
            # Check bounds and obstacles
            if self.is_valid_state(new_state):
                successors.append((new_state, params['cost'], move))
        
        return successors
    
    def is_valid_state(self, state):
        """Check if state is within bounds and not in obstacle"""
        x, y = int(state.x), int(state.y)
        return (0 <= x < ROWS and 0 <= y < COLS and world[x][y] == 0)
    
    def heuristic(self, state1, state2):
        """Manhattan distance heuristic"""
        return abs(state1.x - state2.x) + abs(state1.y - state2.y)
    
    def plan(self, start_pos, goal_pos, start_theta=0):
        """A* search on lattice"""
        start_state = SimpleLatticeState(start_pos[0], start_pos[1], start_theta)
        goal_state = SimpleLatticeState(goal_pos[0], goal_pos[1], 0)  # Goal orientation flexible
        
        open_set = []
        self.node_counter = 0
        heapq.heappush(open_set, (0, 0, self.node_counter, start_state, []))
        self.node_counter += 1
        visited = set()
        
        while open_set:
            f_cost, g_cost, _, current_state, path = heapq.heappop(open_set)
            
            state_key = (round(current_state.x), round(current_state.y), round(current_state.theta, 1))
            if state_key in visited:
                continue
            visited.add(state_key)
            
            # Check if reached goal (position only)
            if (abs(current_state.x - goal_state.x) < 0.1 and 
                abs(current_state.y - goal_state.y) < 0.1):
                return path + [(current_state, 0, 'GOAL')]
            
            # Expand successors
            for next_state, move_cost, move_type in self.get_successors(current_state):
                new_g_cost = g_cost + move_cost
                h_cost = self.heuristic(next_state, goal_state)
                f_cost = new_g_cost + h_cost
                
                next_key = (round(next_state.x), round(next_state.y), round(next_state.theta, 1))
                if next_key not in visited:
                    new_path = path + [(current_state, move_cost, move_type)]
                    heapq.heappush(open_set, (f_cost, new_g_cost, self.node_counter, next_state, new_path))
                    self.node_counter += 1
        
        return []  # No path found

# Initialize Robot  
robot = Robot()
timestep = int(robot.getBasicTimeStep())
robot.step(timestep)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

left_sensor = robot.getDevice("left wheel sensor")
right_sensor = robot.getDevice("right wheel sensor")
left_sensor.enable(timestep)
right_sensor.enable(timestep)

# Movement functions
def rotate_to(angle):
    if abs(angle) < 0.01:
        return
    duration = abs(angle) / (math.pi / 2) * 1.12
    direction = 1 if angle > 0 else -1
    end_time = robot.getTime() + duration
    left_motor.setVelocity(-2.0 * direction)
    right_motor.setVelocity(2.0 * direction)
    while robot.step(timestep) != -1 and robot.getTime() < end_time:
        pass
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

def move_forward():
    initial_left_pos = left_sensor.getValue()
    while robot.step(timestep) != -1:
        current_left_pos = left_sensor.getValue()
        distance = (current_left_pos - initial_left_pos) * WHEEL_RADIUS
        if distance >= TILE_SIZE:
            break
        left_motor.setVelocity(2.0)
        right_motor.setVelocity(2.0)
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

def execute_lattice_path(path):
    """Execute the lattice path"""
    current_theta = 0  # Initial orientation
    
    for i, (state, cost, move) in enumerate(path):
        print(f"Step {i}: Move {move} to ({state.x}, {state.y}, {state.theta:.2f})")
        
        if move == 'S':
            move_forward()
        elif move == 'L':
            rotate_to(math.pi/2)
            current_theta += math.pi/2
        elif move == 'R':
            rotate_to(-math.pi/2)
            current_theta -= math.pi/2
        elif move == 'GOAL':
            print("Reached goal!")
            break

# Main execution
if __name__ == "__main__":
    # Create lattice planner
    lattice = SimpleLattice()
    
    # Plan path
    start = (4, 0)  # row 5, col 1
    goal = (0, 4)   # row 1, col 5
    
    print("Planning lattice path...")
    path = lattice.plan(start, goal, start_theta=0)
    
    if path:
        print(f"Found path with {len(path)} steps:")
        for i, (state, cost, move) in enumerate(path):
            print(f"  {i}: {move} -> ({state.x}, {state.y}, {state.theta:.2f})")
        
        # Execute the path
        execute_lattice_path(path)
    else:
        print("No path found!")