from controller import Robot, Camera, Accelerometer, DistanceSensor, Motor, PositionSensor
import math
import heapq

# Constants
WHEEL_RADIUS = 0.0205
AXLE_LENGTH = 0.052
TILE_SIZE = 0.1

# Map: 0 - free, 1 - obstacle
# (first row is top row in the arena)
world = [
    [1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]
]
ROWS, COLS = len(world), len(world[0])

# Start and goal
start = (4, 0)  # row 5, col 1
goal = (0, 4)   # row 1, col 5

# Directions: (dx, dy), angle
directions = [((0, 1), 0), ((-1, 0), math.pi/2), ((0, -1), math.pi), ((1, 0), -math.pi/2)]

def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(world, start, goal):
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, []))
    visited = set()
    while open_set:
        _, cost, current, path = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        if current == goal:
            return path + [current]
        for (dx, dy), angle in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < ROWS and 0 <= ny < COLS and world[nx][ny] == 0:
                heapq.heappush(open_set, (cost + 1 + heuristic((nx, ny), goal), cost + 1, (nx, ny), path + [current]))
    return []

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
    duration = abs(angle) / (math.pi / 2) * 1.12  # Rough time estimate for 90 deg
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

# Run A*
path = astar(world, start, goal)
print("Planned Path:", path)

# Initial orientation: facing right (0 rad)
direction = (0, 1)

for i in range(1, len(path)):
    current = path[i-1]
    next_pos = path[i]
    dx, dy = next_pos[0] - current[0], next_pos[1] - current[1]
    desired_dir = (dx, dy)
    print(f"Desired direction: {desired_dir}")  # Debug

    for d, angle in directions:
        if d == desired_dir:
            # Get current angle
            current_angle = 0
            for d2, a2 in directions:
                if d2 == direction:
                    current_angle = a2
                    break
            target_angle = angle
            delta = target_angle - current_angle

            # Normalize to [-pi, pi]
            while delta > math.pi:
                delta -= 2 * math.pi
            while delta < -math.pi:
                delta += 2 * math.pi

            print(f"Angle delta to rotate: {delta:.6f} radians")  # Debug

            # Fix: Only rotate if delta is significantly different from 0
            if abs(delta) > 1e-3:
                rotate_to(delta)

            direction = desired_dir
            break

    move_forward()
    print(f"Reached point: {next_pos}")
