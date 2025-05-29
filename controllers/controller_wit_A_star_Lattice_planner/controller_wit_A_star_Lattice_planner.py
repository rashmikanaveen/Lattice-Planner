from controller import Robot, Motor, PositionSensor
import math
import heapq
import numpy as np
from collections import defaultdict

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

class LatticeState:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = self.normalize_angle(theta)

    def normalize_angle(self, angle):
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

    def __repr__(self):
        return f"State({self.x}, {self.y}, {self.theta:.2f})"

class LatticeEdge:
    def __init__(self, from_state, to_state, move_type, cost):
        self.from_state = from_state
        self.to_state = to_state
        self.move_type = move_type
        self.cost = cost

class PrecomputedLatticeGraph:
    def __init__(self, world_map):
        self.world = world_map
        self.rows, self.cols = len(world_map), len(world_map[0])

        # Pre-defined moves
        self.moves = {
            'S': {'cost': 1.0, 'theta_change': 0},
            'L': {'cost': math.pi/2, 'theta_change': math.pi/2},
            'R': {'cost': math.pi/2, 'theta_change': -math.pi/2}
        }

        # Graph storage
        self.states = set()  # All valid states in the lattice
        self.edges = defaultdict(list)  # state -> list of outgoing edges
        self.state_lookup = {}  # (x,y,theta) -> state object

        print("Building complete lattice graph...")
        self._build_complete_lattice()
        print(f"Lattice built: {len(self.states)} states, {sum(len(edges) for edges in self.edges.values())} edges")

    def _build_complete_lattice(self):
        """Pre-generate the entire lattice graph"""

        # Step 1: Generate all possible valid states
        print("Step 1: Generating all valid states...")
        self._generate_all_states()

        # Step 2: Generate all possible edges between states
        print("Step 2: Connecting states with edges...")
        self._generate_all_edges()

    def _generate_all_states(self):
        """Generate all valid (x, y, theta) combinations"""
        # Discrete orientations: 0, π/2, π, -π/2 (0°, 90°, 180°, 270°)
        orientations = [0, math.pi/2, math.pi, -math.pi/2]

        for x in range(self.rows):
            for y in range(self.cols):
                if self.world[x][y] == 0:  # Free space
                    for theta in orientations:
                        state = LatticeState(x, y, theta)
                        self.states.add(state)
                        # Create lookup for fast access
                        key = (round(x), round(y), round(theta, 1))
                        self.state_lookup[key] = state

    def _generate_all_edges(self):
        """Generate all valid transitions between states"""

        for state in self.states:
            # Try each possible move from this state
            for move_type, move_params in self.moves.items():
                successor = self._compute_successor(state, move_type, move_params)

                if successor and successor in self.states:
                    # Create edge
                    edge = LatticeEdge(state, successor, move_type, move_params['cost'])
                    self.edges[state].append(edge)

    def _compute_successor(self, state, move_type, move_params):
        """Compute the successor state for a given move"""
        new_theta = state.theta + move_params['theta_change']

        if move_type == 'S':
            # Forward movement based on current orientation
            if abs(state.theta) < 0.1:  # Facing right (0°)
                new_x, new_y = state.x, state.y + 1
            elif abs(state.theta - math.pi/2) < 0.1:  # Facing up (90°)
                new_x, new_y = state.x - 1, state.y
            elif abs(abs(state.theta) - math.pi) < 0.1:  # Facing left (180°)
                new_x, new_y = state.x, state.y - 1
            elif abs(state.theta + math.pi/2) < 0.1:  # Facing down (270°)
                new_x, new_y = state.x + 1, state.y
            else:
                return None
        else:
            # Turn in place
            new_x, new_y = state.x, state.y

        # Check if successor is valid
        if (0 <= new_x < self.rows and 0 <= new_y < self.cols and
            self.world[new_x][new_y] == 0):
            return LatticeState(new_x, new_y, new_theta)

        return None

    def get_closest_state(self, x, y, theta):
        """Find the closest state in the lattice to given pose"""
        key = (round(x), round(y), round(theta, 1))
        return self.state_lookup.get(key, None)

    def get_neighbors(self, state):
        """Get all neighboring states (successors) of a given state"""
        return self.edges[state]

class LatticePathPlanner:
    def __init__(self, world_map):
        # Pre-build the entire lattice graph
        self.lattice_graph = PrecomputedLatticeGraph(world_map)

    def plan_path(self, start_pose, goal_pose):
        """
        Plan path from start to goal using pre-built lattice

        Args:
            start_pose: (x, y, theta) starting configuration
            goal_pose: (x, y, theta) goal configuration

        Returns:
            List of (state, move_type) pairs representing the path
        """

        # Find closest states in lattice
        start_state = self.lattice_graph.get_closest_state(*start_pose)
        goal_state = self.lattice_graph.get_closest_state(*goal_pose)

        if start_state is None:
            raise ValueError(f"Start pose {start_pose} not in lattice")
        if goal_state is None:
            raise ValueError(f"Goal pose {goal_pose} not in lattice")

        print(f"Planning from {start_state} to {goal_state}")

        # A* search on pre-built graph
        return self._astar_search(start_state, goal_state)

    def _astar_search(self, start_state, goal_state):
        """A* search on the pre-built lattice graph"""

        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, 0, counter, start_state, []))

        visited = set()

        while open_set:
            f_cost, g_cost, _, current_state, path = heapq.heappop(open_set)

            if current_state in visited:
                continue
            visited.add(current_state)

            # Goal check
            if current_state == goal_state:
                return path + [(current_state, 'GOAL')]

            # Expand neighbors using pre-built graph
            for edge in self.lattice_graph.get_neighbors(current_state):
                next_state = edge.to_state

                if next_state not in visited:
                    new_g_cost = g_cost + edge.cost
                    h_cost = self._heuristic(next_state, goal_state)
                    f_cost = new_g_cost + h_cost

                    new_path = path + [(current_state, edge.move_type)]
                    counter += 1
                    heapq.heappush(open_set, (f_cost, new_g_cost, counter, next_state, new_path))

        return []  # No path found

    def _heuristic(self, state1, state2):
        """Manhattan distance heuristic"""
        return abs(state1.x - state2.x) + abs(state1.y - state2.y)

# Robot Controller
class LatticeRobotController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        self.robot.step(self.timestep)

        # Initialize motors and sensors
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.left_sensor = self.robot.getDevice("left wheel sensor")
        self.right_sensor = self.robot.getDevice("right wheel sensor")
        self.left_sensor.enable(self.timestep)
        self.right_sensor.enable(self.timestep)

        # Initialize planner with pre-built lattice
        self.planner = LatticePathPlanner(world)

    def execute_mission(self, start_pose, goal_pose):
        """Execute a mission from start to goal"""
        print(f"Mission: {start_pose} -> {goal_pose}")

        # Plan path using pre-built lattice
        path = self.planner.plan_path(start_pose, goal_pose)

        if not path:
            print("No path found!")
            return False

        print(f"Path found with {len(path)} steps:")
        for i, (state, move) in enumerate(path):
            print(f"  {i}: {move} -> {state}")

        # Execute path
        self._execute_path(path)
        return True

    def _execute_path(self, path):
        """Execute the planned path"""
        for i, (state, move) in enumerate(path):
            print(f"Executing step {i}: {move}")

            if move == 'S':
                self._move_forward()
            elif move == 'L':
                self._rotate(math.pi/2)
            elif move == 'R':
                self._rotate(-math.pi/2)
            elif move == 'GOAL':
                print("Reached goal!")
                break

    def _move_forward(self):
        initial_pos = self.left_sensor.getValue()
        while self.robot.step(self.timestep) != -1:
            current_pos = self.left_sensor.getValue()
            distance = (current_pos - initial_pos) * WHEEL_RADIUS
            if distance >= TILE_SIZE:
                break
            self.left_motor.setVelocity(2.0)
            self.right_motor.setVelocity(2.0)
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

    def _rotate(self, angle):
        if abs(angle) < 0.01:
            return
        duration = abs(angle) / (math.pi / 2) * 1.12
        direction = 1 if angle > 0 else -1
        end_time = self.robot.getTime() + duration
        self.left_motor.setVelocity(-2.0 * direction)
        self.right_motor.setVelocity(2.0 * direction)
        while self.robot.step(self.timestep) != -1 and self.robot.getTime() < end_time:
            pass
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

# Main execution
if __name__ == "__main__":
    controller = LatticeRobotController()

    # Multiple missions with different start/goal pairs
    missions = [
        ((4, 0, 0), (0, 4, 0)),           # Bottom-left to top-right
        ((0, 4, 0), (2, 2, math.pi/2)),   # Top-right to center facing up
        ((2, 2, math.pi/2), (4, 0, math.pi))  # Center to bottom-left facing left
    ]

    for start, goal in missions:
        print(f"\n--- New Mission ---")
        success = controller.execute_mission(start, goal)
        if not success:
            print("Mission failed!")
            break
        print("Mission completed!\n")