#!/usr/bin/env python3
"""
6-DOF Test Bed - Complete Corner Position Calculator
Integrates Arduino control with kinematic transformations
"""

import serial
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

# =============================================================================
# CONFIGURATION - CHANGE THESE FOR EACH TEST
# =============================================================================

# Window dimensions (mm)
WINDOW_WIDTH = 1635.0   # w
WINDOW_HEIGHT = 2580.0  # h

# Window position offsets in bed frame (mm)
X_OFFSET = 1960  # xoff
Y_OFFSET = 215    # yoff   
Z_OFFSET = -0   # zoff

# Arduino serial port
ARDUINO_PORT = '/dev/ttyACM0'  # Change to /dev/ttyACM0 if needed

# =============================================================================
# ROTATION MATRIX FUNCTIONS
# =============================================================================
#18.5 in y
def rotation_x(angle_deg):
    """Roll: Rotation about X axis (NED convention)"""
    theta = np.radians(angle_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(theta), -np.sin(theta)],
        [0, np.sin(theta), np.cos(theta)]
    ])

def rotation_y(angle_deg):
    """Pitch: Rotation about Y axis (NED convention)"""
    theta = np.radians(angle_deg)
    return np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

def rotation_z(angle_deg):
    """Yaw: Rotation about Z axis (NED convention)"""
    theta = np.radians(angle_deg)
    return np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

# =============================================================================
# KINEMATICS CLASS
# =============================================================================

class TestBedKinematics:
    """Handles all kinematic transformations for the 6-DOF test bed"""
    
    def __init__(self):
        """Initialize with your specific kinematic chain parameters"""
        pass
    
    def compute_lidar_position(self, x, y, z, yaw, pitch, roll):
        """
        Compute LiDAR position in bed frame using your kinematic equations
        
        Args:
            x, y, z: Position from rulers (mm)
            yaw, pitch, roll: Servo angles from neutral (degrees)
                            (i.e., servo_angle - 90)
        
        Returns:
            vlidar: LiDAR position in bed frame [x, y, z] (mm)
            R_total: Total rotation matrix (bed → LiDAR)
        """
        # v3dof: Base position (yaw rotation center in bed frame)
        # Your equation: (62.25-x)i + (375-y)j + (-293.38-z)k
        v3dof = np.array([
            62.25 - x,
            375 - y,
            -293.38 - z
        ])
        
        # vyaw_global: Already in global frame from your equation
        # vyaw = -20.7sin(yaw)i + 20.7cos(yaw)j - 44.2k
        vyaw_global = np.array([
            -20.7 * np.sin(np.radians(yaw)),
            20.7 * np.cos(np.radians(yaw)),
            -44.2
        ])
        
        # vpitch: In pitch bracket frame, needs yaw rotation
        # Your equation: 10.58cos(pitch)i - 20j - 10.58sin(pitch)k
        vpitch_local = np.array([
            10.58 * np.cos(np.radians(pitch)),
            -20,
            -10.58 * np.sin(np.radians(pitch))
        ])
        R_yaw = rotation_z(yaw)
        vpitch_global = R_yaw @ vpitch_local
        
        # vroll: In roll bracket frame, needs yaw + pitch rotation
        # Your equation: 15i - 50cos(roll)j - 50sin(roll)k
        vroll_local = np.array([
            15,
            -50 * np.cos(np.radians(roll)),
            -50 * np.sin(np.radians(roll))
        ])
        R_pitch = rotation_y(pitch)
        R_yaw_pitch = R_yaw @ R_pitch
        vroll_global = R_yaw_pitch @ vroll_local
        
        # Total rotation matrix (for corner transformations)
        R_roll = rotation_x(roll)
        R_total = R_yaw @ R_pitch @ R_roll
        
        # Final LiDAR position
        vlidar = v3dof + vyaw_global + vpitch_global + vroll_global
        
        return vlidar, R_total
    
    def compute_window_corners_bed_frame(self, xoff, yoff, zoff, w, h):
        """
        Compute 4 window corners in bed frame using your equations
        
        Args:
            xoff, yoff, zoff: Window position offsets (mm)
            w: Window width (mm)
            h: Window height (mm)
        
        Returns:
            Dictionary of corner positions in bed frame
        """
        # Your corner equations:
        # LR = xoff*i + (490+yoff)*j + zoff*k
        # LL = xoff*i + (490+yoff-w)*j + zoff*k
        # UR = xoff*i + (490+yoff)*j + (zoff-h)*k
        # UL = xoff*i + (490+yoff-w)*j + (zoff-h)*k
        
        corners = {
            'LR': np.array([xoff, 490 + yoff, zoff]),           # Lower Right
            'LL': np.array([xoff, 490 + yoff - w, zoff]),       # Lower Left
            'UR': np.array([xoff, 490 + yoff, zoff - h]),       # Upper Right
            'UL': np.array([xoff, 490 + yoff - w, zoff - h])    # Upper Left
        }
        
        return corners
    
    def transform_corners_to_lidar_frame(self, corners_bed, vlidar, R_total):
        """
        Transform corners from bed frame to LiDAR frame
        
        Args:
            corners_bed: Dictionary of corners in bed frame
            vlidar: LiDAR position in bed frame
            R_total: Rotation matrix (bed → LiDAR orientation)
        
        Returns:
            Dictionary of corners in LiDAR frame
        """
        corners_lidar = {}
        
        # Inverse rotation: R^(-1) = R^T for rotation matrices
        R_lidar_to_bed = R_total.T
        
        for name, corner_bed in corners_bed.items():
            # Step 1: Translate to LiDAR-centered coordinates
            corner_relative = corner_bed - vlidar
            
            # Step 2: Rotate into LiDAR's frame
            corner_lidar = R_lidar_to_bed @ corner_relative
            
            corners_lidar[name] = corner_lidar
        
        return corners_lidar
    
    def compute_plane_equation(self, corners_lidar):
        """
        Compute plane equation from 4 corners using least squares fit
        
        Plane equation: ax + by + cz + d = 0
        Or in normal form: n·(r - r0) = 0
        
        Args:
            corners_lidar: Dictionary of 4 corners in LiDAR frame
        
        Returns:
            normal: Normal vector [a, b, c] (unit vector)
            d: Distance parameter
            equation_str: Human-readable equation string
        """
        # Extract corner positions
        points = np.array([
            corners_lidar['LR'],
            corners_lidar['LL'],
            corners_lidar['UR'],
            corners_lidar['UL']
        ])
        
        # Use 3 points to define plane (4th for verification)
        # Take LL as reference point
        p0 = points[1]  # LL
        p1 = points[0]  # LR
        p2 = points[2]  # UR
        
        # Two vectors in the plane
        v1 = p1 - p0
        v2 = p2 - p0
        
        # Normal vector (cross product)
        normal = np.cross(v1, v2)
        
        # Normalize to unit vector
        normal = normal / np.linalg.norm(normal)
        
        # Compute d: ax0 + by0 + cz0 + d = 0 → d = -(ax0 + by0 + cz0)
        d = -np.dot(normal, p0)
        
        # Create equation string
        equation_str = f"{normal[0]:.4f}x + {normal[1]:.4f}y + {normal[2]:.4f}z + {d:.4f} = 0"
        
        # Also compute distance from origin to plane
        distance_to_origin = abs(d) / np.linalg.norm(normal[:3])
        
        return normal, d, equation_str, distance_to_origin

# =============================================================================
# ARDUINO CONTROLLER CLASS
# =============================================================================

class TestBedController:
    """Controls Arduino servos via USB serial"""
    
    def __init__(self, port=ARDUINO_PORT, baudrate=115200, timeout=1):
        """
        Initialize connection to Arduino
        
        Args:
            port: Serial port (usually /dev/ttyUSB0 or /dev/ttyACM0)
            baudrate: Must match Arduino (115200)
            timeout: Read timeout in seconds
        """
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)  # Wait for Arduino to reset after connection
            print(f"✓ Connected to Arduino on {port}")
            
            # Read and print startup messages
            time.sleep(0.5)
            while self.ser.in_waiting:
                print(self.ser.readline().decode('utf-8', errors='ignore').strip())
                
        except serial.SerialException as e:
            print(f"✗ Error connecting to {port}: {e}")
            print("  Try: ls /dev/tty* to find correct port")
            raise
    
    def send_command(self, command, verbose=False):
        """Send command to Arduino and read response"""
        if not command.endswith('\n'):
            command += '\n'
        
        self.ser.write(command.encode('utf-8'))
        time.sleep(0.1)
        
        response = []
        while self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                response.append(line)
                if verbose:
                    print(f"  Arduino: {line}")
        
        return response
    
    def set_angles(self, pitch, yaw, roll):
        """
        Set all three servo angles at once
        
        Args:
            pitch, yaw, roll: Angles in degrees from neutral
                            (will be converted to 0-180 servo range)
        """
        # Convert from "angles from neutral" to servo angles (0-180, where 90=neutral)
        servo_pitch = 90 - pitch-5
        servo_yaw = 90 - yaw
        servo_roll = roll + 90
        
        # Clamp to valid range
        servo_pitch = max(0, min(180, servo_pitch))
        servo_yaw = max(0, min(180, servo_yaw))
        servo_roll = max(0, min(180, servo_roll))
        
        return self.send_command(f'A{servo_pitch} {servo_yaw} {servo_roll}')
    
    def neutral(self):
        """Move all axes to neutral (90 degrees)"""
        return self.send_command('N')
    
    def close(self):
        """Close serial connection"""
        self.ser.close()
        print("Arduino connection closed")

# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class TestBedApplication:
    """Main application integrating Arduino control and kinematics"""
    
    def __init__(self):
        """Initialize application"""
        print("=" * 70)
        print("6-DOF TEST BED - WINDOW CORNER POSITION CALCULATOR")
        print("=" * 70)
        print()
        
        # Initialize components
        self.arduino = TestBedController(port=ARDUINO_PORT)
        self.kinematics = TestBedKinematics()
        
        # Display configuration
        self.print_configuration()
    
    def print_configuration(self):
        """Display current configuration"""
        print("Current Configuration:")
        print(f"  Window: {WINDOW_WIDTH} mm × {WINDOW_HEIGHT} mm")
        print(f"  Offset: X={X_OFFSET}, Y={Y_OFFSET}, Z={Z_OFFSET} mm")
        print(f"  Arduino: {ARDUINO_PORT}")
        print()
    
    def get_user_input(self):
        """
        Get position and angle inputs from user
        
        Returns:
            x, y, z, yaw, pitch, roll (all floats)
        """
        print("-" * 70)
        print("Enter test bed state:")
        print()
        
        # Linear positions from rulers
        x = float(input("  X (mm from ruler): "))
        y = float(input("  Y (mm from ruler): "))
        z = float(input("  Z (mm from ruler): "))
        
        # Angular positions (from neutral)
        print()
        print("  Angles (degrees from neutral, where neutral=0°):")
        yaw = float(input("    Yaw:   "))
        pitch = float(input("    Pitch: "))
        roll = float(input("    Roll:  "))
        
        return x, y, z, yaw, pitch, roll
    
    def process_measurement(self, x, y, z, yaw, pitch, roll):
        """
        Process a single measurement: move servos and compute corners
        
        Args:
            x, y, z: Linear positions (mm)
            yaw, pitch, roll: Angles from neutral (degrees)
        """
        print()
        print("=" * 70)
        print("PROCESSING MEASUREMENT")
        print("=" * 70)
        
        # Step 1: Send angles to servos
        print("\n[1/4] Sending angles to servos...")
        self.arduino.set_angles(pitch, yaw, roll)
        time.sleep(1)  # Give servos time to move
        print("  ✓ Servos positioned")
        
        # Step 2: Compute LiDAR position
        print("\n[2/4] Computing LiDAR position...")
        vlidar, R_total = self.kinematics.compute_lidar_position(
            x, y, z, yaw, pitch, roll
        )
        print(f"  LiDAR position in bed frame:")
        print(f"    X: {vlidar[0]:8.2f} mm (North)")
        print(f"    Y: {vlidar[1]:8.2f} mm (East)")
        print(f"    Z: {vlidar[2]:8.2f} mm (Down)")
        
        # Step 3: Compute window corners
        print("\n[3/4] Computing window corners...")
        
        # Corners in bed frame
        corners_bed = self.kinematics.compute_window_corners_bed_frame(
            X_OFFSET, Y_OFFSET, Z_OFFSET, WINDOW_WIDTH, WINDOW_HEIGHT
        )
        
        # Transform to LiDAR frame
        corners_lidar = self.kinematics.transform_corners_to_lidar_frame(
            corners_bed, vlidar, R_total
        )
        
        print("  Window corners in LiDAR frame (NED):")
        for name, corner in corners_lidar.items():
            distance = np.linalg.norm(corner)
            print(f"    {name}: [{corner[0]:7.2f}, {corner[1]:7.2f}, {corner[2]:7.2f}] mm"
                  f"  (dist: {distance/1000:.3f} m)")
        
        # Step 4: Compute plane equation
        print("\n[4/4] Computing plane equation...")
        normal, d, equation_str, dist_origin = self.kinematics.compute_plane_equation(
            corners_lidar
        )
        
        print(f"  Plane equation: {equation_str}")
        print(f"  Normal vector: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
        print(f"  Distance from LiDAR origin: {dist_origin:.2f} mm ({dist_origin/1000:.3f} m)")
        
        # Return results for potential saving
        return {
            'input': {'x': x, 'y': y, 'z': z, 'yaw': yaw, 'pitch': pitch, 'roll': roll},
            'vlidar': vlidar,
            'corners_bed': corners_bed,
            'corners_lidar': corners_lidar,
            'plane': {'normal': normal, 'd': d, 'equation': equation_str}
        }
    
    def save_results(self, results, filename):
        """Save results to text file"""
        with open(filename, 'w') as f:
            f.write("6-DOF Test Bed - Measurement Results\n")
            f.write("=" * 70 + "\n\n")
            
            # Input state
            inp = results['input']
            f.write("Input State:\n")
            f.write(f"  Position: X={inp['x']}, Y={inp['y']}, Z={inp['z']} mm\n")
            f.write(f"  Angles: Yaw={inp['yaw']}°, Pitch={inp['pitch']}°, Roll={inp['roll']}°\n\n")
            
            # LiDAR position
            v = results['vlidar']
            f.write("LiDAR Position (Bed Frame):\n")
            f.write(f"  [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}] mm\n\n")
            
            # Corners in LiDAR frame
            f.write("Window Corners (LiDAR Frame, NED):\n")
            for name, corner in results['corners_lidar'].items():
                f.write(f"  {name}: [{corner[0]:.2f}, {corner[1]:.2f}, {corner[2]:.2f}] mm\n")
            f.write("\n")
            
            # Plane equation
            f.write("Plane Equation:\n")
            f.write(f"  {results['plane']['equation']}\n")
            f.write(f"  Normal: {results['plane']['normal']}\n")
        
        print(f"\n  ✓ Results saved to {filename}")
    
    def run(self):
        """Main application loop"""
        print("Ready to process measurements!")
        print("Commands: 'q' = quit, 's' = save last result, 'n' = move to neutral")
        print()
        
        last_results = None
        
        while True:
            try:
                # Get command or input
                cmd = input("\nPress Enter to start measurement (or command): ").strip().lower()
                
                if cmd == 'q':
                    print("\nExiting...")
                    break
                
                elif cmd == 'n':
                    print("Moving servos to neutral...")
                    self.arduino.neutral()
                    time.sleep(1)
                    continue
                
                elif cmd == 's':
                    if last_results:
                        filename = input("Filename (without .txt): ").strip()
                        if not filename:
                            filename = f"measurement_{int(time.time())}"
                        if not filename.endswith('.txt'):
                            filename += '.txt'
                        self.save_results(last_results, filename)
                    else:
                        print("No results to save yet!")
                    continue
                
                elif cmd != '':
                    print("Unknown command. Use: Enter (measure), 'n' (neutral), 's' (save), 'q' (quit)")
                    continue
                
                # Get measurements
                x, y, z, yaw, pitch, roll = self.get_user_input()
                
                # Process and display results
                last_results = self.process_measurement(x, y, z, yaw, pitch, roll)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except ValueError as e:
                print(f"\n✗ Invalid input: {e}")
                continue
            except Exception as e:
                print(f"\n✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Cleanup
        self.arduino.close()
        print("\nGoodbye!")

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = TestBedApplication()
    app.run()