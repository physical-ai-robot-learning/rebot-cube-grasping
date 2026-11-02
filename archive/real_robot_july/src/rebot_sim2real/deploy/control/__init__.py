"""Control subpackage: per-tick control loop + FSM (per-mode step) + dashboard command routing + recovery.

Layout:
  - context      : FSM state as per-concern dataclasses (data-only)
  - control_loop : per-tick preamble + mode dispatch + while loop  (P2.3)
  - modes        : per-mode step functions (run/align/follow/approach/lift/fk_touch/grip/autograsp)  (P2.3)
  - commands     : dashboard command routing  (P2.3)
  - recovery     : return to REST / estop / z-correction  (P2.3)
"""
