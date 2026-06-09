#!/usr/bin/env python3
"""
6-DOF Test Bed - MULTI-WINDOW Corner Position Calculator
========================================================
Extends the single-window test_bed (1).py to a GRID of windows.

The facade geometry is entered ONCE (interactive, at program start) using a
neighbor-relative scheme:
  - rows x cols grid
  - an inclusion mask (which cells actually have a window)
  - each window's width & height (entered in a zigzag column sweep)
  - each window's position pinned by ONE corner:
        * the first window  -> absolute, vs the testbed corner
        * later windows      -> relative to an already-known neighbor's corner
        * windows with no known neighbor (islands / diagonals) -> absolute fallback

Then the measurement LOOP is the same as before: each scan you enter only the
6 LiDAR-pose numbers (x,y,z,yaw,pitch,roll) and the code outputs EVERY window's
4 corners in the LiDAR frame, arranged in an R x C table.

The kinematics (compute_lidar_position / transform_corners_to_lidar_frame /
compute_plane_equation) are copied verbatim from the single-window version.
"""

import time
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================
ARDUINO_PORT = '/dev/ttyACM0'  # Raspberry Pi / Linux serial port
TESTBED_Y_CONST = 490.0        # fixed testbed-origin->window offset along Y,
                               # exactly as in the single-window code
                               # (LR corner y = TESTBED_Y_CONST + yoff)

# =============================================================================
# ROTATION MATRIX FUNCTIONS  (unchanged, NED convention)
# =============================================================================
def rotation_x(angle_deg):
    """Roll: rotation about X axis."""
    t = np.radians(angle_deg)
    return np.array([[1, 0, 0],
                     [0, np.cos(t), -np.sin(t)],
                     [0, np.sin(t),  np.cos(t)]])

def rotation_y(angle_deg):
    """Pitch: rotation about Y axis."""
    t = np.radians(angle_deg)
    return np.array([[ np.cos(t), 0, np.sin(t)],
                     [0, 1, 0],
                     [-np.sin(t), 0, np.cos(t)]])

def rotation_z(angle_deg):
    """Yaw: rotation about Z axis."""
    t = np.radians(angle_deg)
    return np.array([[np.cos(t), -np.sin(t), 0],
                     [np.sin(t),  np.cos(t), 0],
                     [0, 0, 1]])

# =============================================================================
# KINEMATICS  (copied verbatim from the single-window version)
# =============================================================================
class TestBedKinematics:
    def compute_lidar_position(self, x, y, z, yaw, pitch, roll):
        """Compute LiDAR position (bed frame) and total rotation matrix."""
        v3dof = np.array([62.25 - x, 375 - y, -293.38 - z])
        vyaw_global = np.array([-20.7 * np.sin(np.radians(yaw)),
                                 20.7 * np.cos(np.radians(yaw)),
                                -44.2])
        vpitch_local = np.array([10.58 * np.cos(np.radians(pitch)),
                                 -20,
                                -10.58 * np.sin(np.radians(pitch))])
        R_yaw = rotation_z(yaw)
        vpitch_global = R_yaw @ vpitch_local

        vroll_local = np.array([15,
                                -50 * np.cos(np.radians(roll)),
                                -50 * np.sin(np.radians(roll))])
        R_pitch = rotation_y(pitch)
        R_yaw_pitch = R_yaw @ R_pitch
        vroll_global = R_yaw_pitch @ vroll_local

        R_roll = rotation_x(roll)
        R_total = R_yaw @ R_pitch @ R_roll

        vlidar = v3dof + vyaw_global + vpitch_global + vroll_global
        return vlidar, R_total

    def transform_corners_to_lidar_frame(self, corners_bed, vlidar, R_total):
        """Translate (subtract LiDAR pos) then rotate (R^T) into the LiDAR frame."""
        R_lidar_to_bed = R_total.T
        out = {}
        for name, corner_bed in corners_bed.items():
            out[name] = R_lidar_to_bed @ (corner_bed - vlidar)
        return out

    def compute_plane_equation(self, corners_lidar):
        """Fit plane ax+by+cz+d=0 from a window's 4 corners (in LiDAR frame)."""
        p = np.array([corners_lidar['LR'], corners_lidar['LL'],
                      corners_lidar['UR'], corners_lidar['UL']])
        p0, p1, p2 = p[1], p[0], p[2]          # LL, LR, UR
        normal = np.cross(p1 - p0, p2 - p0)
        normal = normal / np.linalg.norm(normal)
        d = -np.dot(normal, p0)
        eq = f"{normal[0]:.4f}x + {normal[1]:.4f}y + {normal[2]:.4f}z + {d:.4f} = 0"
        dist = abs(d) / np.linalg.norm(normal[:3])
        return normal, d, eq, dist

# =============================================================================
# ARDUINO CONTROLLER  (serial imported lazily so geometry is testable without it)
# =============================================================================
class TestBedController:
    def __init__(self, port=ARDUINO_PORT, baudrate=115200, timeout=1):
        import serial                          # lazy import
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(2)
            print(f"  Connected to Arduino on {port}")
            time.sleep(0.5)
            while self.ser.in_waiting:
                print(self.ser.readline().decode('utf-8', errors='ignore').strip())
        except serial.SerialException as e:
            print(f"  Error connecting to {port}: {e}")
            raise

    def send_command(self, command, verbose=False):
        if not command.endswith('\n'):
            command += '\n'
        self.ser.write(command.encode('utf-8'))
        time.sleep(0.1)
        resp = []
        while self.ser.in_waiting:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                resp.append(line)
                if verbose:
                    print(f"  Arduino: {line}")
        return resp

    def set_angles(self, pitch, yaw, roll):
        servo_pitch = max(0, min(180, 90 - pitch - 5))   # -5 = pitch calibration
        servo_yaw   = max(0, min(180, 90 - yaw))
        servo_roll  = max(0, min(180, roll + 90))
        return self.send_command(f'A{servo_pitch} {servo_yaw} {servo_roll}')

    def neutral(self):
        return self.send_command('N')

    def close(self):
        self.ser.close()
        print("  Arduino connection closed")

# =============================================================================
# FACADE GRID GEOMETRY  (pure functions - no I/O, fully unit-testable)
# =============================================================================
# Row index r: 1 = TOP, R = BOTTOM.   Column index c: 1 = LEFT, C = RIGHT.

# Where a neighbor sits relative to the new cell -> (dr, dc) to reach it.
NEIGHBOR_OFFSETS = {'below': (1, 0), 'above': (-1, 0),
                    'left': (0, -1), 'right': (0, 1)}

# Which corner of the NEW window is pinned, and to which corner of the NEIGHBOR.
# Derived directly from the user's 4 rules:
#   neighbor below  -> new.LR vs neighbor.UR
#   neighbor above  -> new.UR vs neighbor.LR
#   neighbor left   -> new.LL vs neighbor.LR
#   neighbor right  -> new.LR vs neighbor.LL
NEIGHBOR_RULES = {'below': ('LR', 'UR'), 'above': ('UR', 'LR'),
                  'left':  ('LL', 'LR'), 'right': ('LR', 'LL')}

# Priority for choosing which known neighbor to reference (matches zigzag order).
NEIGHBOR_PRIORITY = ['below', 'above', 'right', 'left']


def zigzag_cells(R, C):
    """Yield (r,c) in a boustrophedon column sweep starting at the RIGHTMOST
    column going bottom->top, then next column top->bottom, alternating."""
    order = []
    going_up = True                                   # rightmost column: bottom->top
    for c in range(C, 0, -1):
        rows = range(R, 0, -1) if going_up else range(1, R + 1)
        for r in rows:
            order.append((r, c))
        going_up = not going_up
    return order


def build_window_corners(ref_name, ref_xyz, w, h):
    """Given ONE known corner (by name) + width/height, return all 4 corners
    in the bed frame. NED: right=+Y, left=-Y, up=-Z, down=+Z."""
    x, y, z = float(ref_xyz[0]), float(ref_xyz[1]), float(ref_xyz[2])
    if   ref_name == 'LR': y_R, z_B = y,     z          # right edge, bottom edge
    elif ref_name == 'LL': y_R, z_B = y + w, z
    elif ref_name == 'UR': y_R, z_B = y,     z + h
    elif ref_name == 'UL': y_R, z_B = y + w, z + h
    else: raise ValueError(f"bad corner name {ref_name}")
    return {
        'LR': np.array([x, y_R,     z_B]),
        'LL': np.array([x, y_R - w, z_B]),
        'UR': np.array([x, y_R,     z_B - h]),
        'UL': np.array([x, y_R - w, z_B - h]),
    }


def find_known_neighbor(r, c, known):
    """Return (direction, (nr,nc)) of the first already-computed neighbor, else None."""
    for d in NEIGHBOR_PRIORITY:
        dr, dc = NEIGHBOR_OFFSETS[d]
        if (r + dr, c + dc) in known:
            return d, (r + dr, c + dc)
    return None


def build_facade(R, C, mask, dims, xoff, ask_abs, ask_rel):
    """Walk the grid in zigzag order and compute every included window's 4
    corners in the bed frame.

    ask_abs(r,c) -> (yoff, zoff)        absolute bottom-right corner vs testbed
    ask_rel(r,c,direction,neighbor,new_name,nb_name) -> (dy, dz)   relative offset
    """
    corners_by_cell = {}
    for (r, c) in zigzag_cells(R, C):
        if not mask.get((r, c)):
            continue
        w, h = dims[(r, c)]
        nb = find_known_neighbor(r, c, corners_by_cell)
        if nb is None:                                    # anchor / island -> absolute
            yoff, zoff = ask_abs(r, c)
            ref_name = 'LR'
            ref_xyz = np.array([xoff, TESTBED_Y_CONST + yoff, zoff])
        else:                                             # relative to a known neighbor
            direction, (nr, nc) = nb
            new_name, nb_name = NEIGHBOR_RULES[direction]
            nb_corner = corners_by_cell[(nr, nc)][nb_name]
            dy, dz = ask_rel(r, c, direction, (nr, nc), new_name, nb_name)
            ref_name = new_name
            ref_xyz = nb_corner + np.array([0.0, dy, dz])
        corners_by_cell[(r, c)] = build_window_corners(ref_name, ref_xyz, w, h)
    return corners_by_cell

# =============================================================================
# SMALL INPUT HELPERS
# =============================================================================
def ask_int(prompt):
    while True:
        try:
            v = int(input(prompt))
            if v <= 0:
                print("    Enter a positive integer."); continue
            return v
        except ValueError:
            print("    Please enter an integer.")

def ask_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("    Please enter a number.")

def ask_yesno(prompt):
    while True:
        a = input(prompt).strip().lower()
        if a in ('y', 'yes'): return True
        if a in ('n', 'no'):  return False
        print("    Please answer y or n.")

# =============================================================================
# MAIN APPLICATION
# =============================================================================
class MultiWindowApp:
    def __init__(self):
        print("=" * 72)
        print("6-DOF TEST BED - MULTI-WINDOW CORNER CALCULATOR")
        print("=" * 72)
        self.kin = TestBedKinematics()
        self.arduino = TestBedController(port=ARDUINO_PORT)
        # facade state (filled by define_facade)
        self.R = self.C = 0
        self.mask = {}
        self.corners_by_cell = {}

    # ----- one-time facade definition -------------------------------------
    def define_facade(self):
        print("\n" + "-" * 72)
        print("DEFINE THE FACADE (entered once)")
        print("-" * 72)

        R = ask_int("Number of rows: ")
        C = ask_int("Number of columns: ")

        print("\nMark which cells contain a window (r1 = top row, c1 = left col):")
        mask = {}
        for r in range(1, R + 1):
            for c in range(1, C + 1):
                mask[(r, c)] = ask_yesno(f"  r{r}c{c} included? (y/n): ")
        if not any(mask.values()):
            raise ValueError("No windows selected.")

        print("\nEnter WIDTH and HEIGHT of each window (mm), zigzag order:")
        dims = {}
        for (r, c) in zigzag_cells(R, C):
            if not mask[(r, c)]:
                continue
            w = ask_float(f"  r{r}c{c} width  (mm): ")
            h = ask_float(f"  r{r}c{c} height (mm): ")
            dims[(r, c)] = (w, h)

        print("\nAll window corners share the same X (facade depth).")
        xoff = ask_float("  Facade X offset (xoff, mm): ")

        print("\nNow the POSITIONS (zigzag order).")
        print("Sign convention (bed frame): Y+ = right, Z+ = down.")

        def ask_abs(r, c):
            print(f"\n  r{r}c{c}: no known neighbor -> ABSOLUTE bottom-right (LR) "
                  f"corner vs testbed:")
            yoff = ask_float("     y offset (yoff, mm): ")
            zoff = ask_float("     z offset (zoff, mm): ")
            return yoff, zoff

        def ask_rel(r, c, direction, neighbor, new_name, nb_name):
            nr, nc = neighbor
            hint = {'below': "(usually Z- = up)", 'above': "(usually Z+ = down)",
                    'left':  "(usually Y- = left)", 'right': "(usually Y+ = right)"}[direction]
            print(f"\n  r{r}c{c}: neighbor r{nr}c{nc} is {direction}. Enter this "
                  f"window's {new_name} corner")
            print(f"     relative to r{nr}c{nc}'s {nb_name} corner {hint}:")
            dy = ask_float("     dy (right +, left -, mm): ")
            dz = ask_float("     dz (down +, up -, mm): ")
            return dy, dz

        self.R, self.C, self.mask = R, C, mask
        self.corners_by_cell = build_facade(R, C, mask, dims, xoff, ask_abs, ask_rel)
        self._print_facade_summary()

    def _print_facade_summary(self):
        print("\n  Facade built. Window bottom-right (LR) corners in bed frame:")
        for r in range(1, self.R + 1):
            for c in range(1, self.C + 1):
                if (r, c) in self.corners_by_cell:
                    lr = self.corners_by_cell[(r, c)]['LR']
                    print(f"    r{r}c{c}: LR = [{lr[0]:8.1f}, {lr[1]:8.1f}, {lr[2]:8.1f}]")

    # ----- per-scan pose input --------------------------------------------
    def get_pose(self):
        print("\nEnter LiDAR pose:")
        x = ask_float("  X (mm from ruler): ")
        y = ask_float("  Y (mm from ruler): ")
        z = ask_float("  Z (mm from ruler): ")
        print("  Angles (deg from neutral):")
        yaw   = ask_float("    Yaw:   ")
        pitch = ask_float("    Pitch: ")
        roll  = ask_float("    Roll:  ")
        return x, y, z, yaw, pitch, roll

    # ----- one measurement ------------------------------------------------
    def process_measurement(self, x, y, z, yaw, pitch, roll):
        print("\n" + "=" * 72)
        print("PROCESSING MEASUREMENT")
        print("=" * 72)

        print("\n[1/3] Sending angles to servos...")
        self.arduino.set_angles(pitch, yaw, roll)
        time.sleep(1)

        print("[2/3] Computing LiDAR pose...")
        vlidar, R_total = self.kin.compute_lidar_position(x, y, z, yaw, pitch, roll)
        print(f"  LiDAR in bed frame: [{vlidar[0]:.1f}, {vlidar[1]:.1f}, {vlidar[2]:.1f}] mm")

        print("[3/3] Transforming every window into the LiDAR frame...")
        table = {}                                   # (r,c) -> 4 corners in LiDAR frame
        for cell, corners_bed in self.corners_by_cell.items():
            table[cell] = self.kin.transform_corners_to_lidar_frame(
                corners_bed, vlidar, R_total)

        self._print_table(table)

        # one facade plane (all windows coplanar) from any window
        first = next(iter(table.values()))
        normal, d, eq, dist = self.kin.compute_plane_equation(first)
        print(f"\n  Facade plane (LiDAR frame): {eq}")
        print(f"  Distance LiDAR -> facade: {dist:.1f} mm ({dist/1000:.3f} m)")

        return {'pose': (x, y, z, yaw, pitch, roll), 'vlidar': vlidar,
                'table': table, 'plane': eq}

    def _print_table(self, table):
        print("\n  Window corners in LiDAR frame (mm), by grid cell:")
        for r in range(1, self.R + 1):
            for c in range(1, self.C + 1):
                if (r, c) not in table:
                    continue
                print(f"    r{r}c{c}:")
                for name in ('UL', 'UR', 'LL', 'LR'):
                    p = table[(r, c)][name]
                    dist = np.linalg.norm(p)
                    print(f"        {name}: [{p[0]:8.1f}, {p[1]:8.1f}, {p[2]:8.1f}]"
                          f"  (dist {dist/1000:.3f} m)")

    def save_results(self, results, filename):
        if not filename.endswith('.txt'):
            filename += '.txt'
        with open(filename, 'w') as f:
            f.write("6-DOF Test Bed - Multi-Window Measurement\n")
            f.write("=" * 72 + "\n\n")
            p = results['pose']
            f.write(f"Pose: X={p[0]} Y={p[1]} Z={p[2]} | "
                    f"Yaw={p[3]} Pitch={p[4]} Roll={p[5]}\n")
            v = results['vlidar']
            f.write(f"LiDAR (bed frame): [{v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f}]\n\n")
            f.write("Window corners in LiDAR frame (mm):\n")
            for (r, c), corners in sorted(results['table'].items()):
                f.write(f"  r{r}c{c}:\n")
                for name in ('UL', 'UR', 'LL', 'LR'):
                    q = corners[name]
                    f.write(f"    {name}: [{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}]\n")
            f.write(f"\nFacade plane: {results['plane']}\n")
        print(f"  Saved to {filename}")

    # ----- main loop ------------------------------------------------------
    def run(self):
        self.define_facade()
        print("\n" + "-" * 72)
        print("Ready. Commands:  Enter = measure | n = neutral | s = save | q = quit")
        last = None
        while True:
            try:
                cmd = input("\n> ").strip().lower()
                if cmd == 'q':
                    break
                elif cmd == 'n':
                    self.arduino.neutral(); time.sleep(1); continue
                elif cmd == 's':
                    if last:
                        fn = input("  filename: ").strip() or f"measurement_{int(time.time())}"
                        self.save_results(last, fn)
                    else:
                        print("  Nothing measured yet.")
                    continue
                elif cmd != '':
                    print("  Unknown command."); continue
                pose = self.get_pose()
                last = self.process_measurement(*pose)
            except KeyboardInterrupt:
                break
            except ValueError as e:
                print(f"  Invalid input: {e}")
        self.arduino.close()
        print("Goodbye!")


if __name__ == "__main__":
    MultiWindowApp().run()
