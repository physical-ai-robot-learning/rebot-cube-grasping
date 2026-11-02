"""reBot real-robot deployment (deploy) runtime package.

Package that splits the old deploy/debug_deepreach.py (monolith, ~2000 lines) by concern:
  - blackboard        : the only shared state between threads (6 sections)
  - io/               : CAN bus hardware (robot_io) + IO thread (io_thread)
  - vision/           : cameras, detection (OWLv2/HSV), front-gripper Z, annotation (perception_thread etc.)
  - kinematics        : pure FK/IK math (wrist projection, gimbal look-at, danger check)
  - main_dashboard    : HTTP dashboard for real-robot operation (embedded; run as a thread by main.py)
  - control/(P2)       : control loop + FSM + command routing
Entry point = deploy/main.py (P3).
"""
