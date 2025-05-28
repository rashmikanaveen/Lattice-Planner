# Copyright 1996-2024 Cyberbotics Ltd.
# Licensed under the Apache License, Version 2.0

from controller import Robot, Camera, Accelerometer, DistanceSensor, Motor, PositionSensor
import math

WHEEL_RADIUS = 0.02
AXLE_LENGTH = 0.052
RANGE = 1024 / 2

# Braitenberg coefficients
braitenberg_coefficients = [
    [0.942, -0.22], [0.63, -0.1], [0.5, -0.06], [-0.06, -0.06],
    [-0.06, -0.06], [-0.06, 0.5], [-0.19, 0.63], [-0.13, 0.942]
]

def compute_odometry(left_position, right_position):
    dl = left_position * WHEEL_RADIUS
    dr = right_position * WHEEL_RADIUS
    da = (dr - dl) / AXLE_LENGTH
    print(f"Estimated distance covered by left wheel: {dl:.4f} m")
    print(f"Estimated distance covered by right wheel: {dr:.4f} m")
    print(f"Estimated change of orientation: {da:.4f} rad")

# Initialize robot
robot = Robot()

# Determine time step
model = robot.getModel()
if model == "GCtronic e-puck2":
    print("e-puck2 robot")
    time_step = 64
    camera_time_step = 64
else:
    print("e-puck robot")
    time_step = 256
    camera_time_step = 1024

# Devices
camera = robot.getDevice("camera")
camera.enable(camera_time_step)

accelerometer = robot.getDevice("accelerometer")
accelerometer.enable(time_step)

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

left_position_sensor = robot.getDevice("left wheel sensor")
right_position_sensor = robot.getDevice("right wheel sensor")
left_position_sensor.enable(time_step)
right_position_sensor.enable(time_step)

# Distance sensors
ps = []
for i in range(8):
    sensor = robot.getDevice(f"ps{i}")
    sensor.enable(time_step)
    ps.append(sensor)

# Main loop
while robot.step(time_step) != -1:
    # Read distance sensors
    sensor_values = [s.getValue() for s in ps]

    # Accelerometer readings
    acc = accelerometer.getValues()
    print(f"Accelerometer values: {acc[0]:.2f}, {acc[1]:.2f}, {acc[2]:.2f}")

    # Compute odometry
    compute_odometry(left_position_sensor.getValue(), right_position_sensor.getValue())

    # Compute speeds using Braitenberg coefficients
    speed = [0.0, 0.0]
    for i in range(2):
        for j in range(8):
            speed[i] += braitenberg_coefficients[j][i] * (1.0 - sensor_values[j] / RANGE)

    # Set wheel velocities
    left_motor.setVelocity(speed[0])
    right_motor.setVelocity(speed[1])
