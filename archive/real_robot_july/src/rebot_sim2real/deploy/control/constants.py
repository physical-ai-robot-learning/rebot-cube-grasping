"""Constants shared by the control loop/modes (extracted from module constants 47-84 of the old debug_deepreach.py).

Values are HW tuning results and are preserved as-is (the refactor only moved their location).
"""

# ---- Z / descent ----
GRASP_CLEAR = 0.005          # same as the deep_reach_step grasp_clear default; descent stops at grasp_clear above cube_top_z
Z_BIAS_MAX = 0.10            # (auto/manual) descent-depth safety clamp — so a bad reading cannot over-descend
FK_BIAS_TAU = 0.4            # auto-z fk_bias EMA time constant (s), rate-independent (alpha=1-exp(-loop_dt/tau)) → smoothness independent of control_hz

# ---- yaw latch ----
YAW_LATCH_Z = (0.05, 0.20)   # z_gap_io band (m, above the cube) where wrist yaw is trustworthy. RUN latches/freezes yaw in this band →
#                              consumed at the close Align pose (there seg minArea grabs the whole close blob and collapses to 0/90°)

# ---- approach look-at ----
LOOKAT_RISE_TOL = 0.015      # approach: 'finished rising to hover' tolerance before arming the look-at descent ramp (m, rise-first)

# ---- Align / auto-grasp ----
ALIGN_DWELL_MAX = 80              # Align auto-center: max J4-servo loops before J6 roll (framing stall guard)
AUTOGRASP_ALIGN_TIMEOUT = 12.0   # --autograsp-yaw: max time (s) for J6 yaw + re-center before descent. Generous enough to hold the long yaw
#                                  accumulation window (below) (slow alignment is OK). Abort on exceed (convergence failure).
AG_ALIGN_DWELL = 0.4             # --autograsp-yaw: (yaw reached + centered + J5 unsaturated) must be stable continuously for this time (s) before descending (prevents momentary blips)
# ⭐ CONSERVATIVE YAW GATE (docs/33): live seg edge-voting yaw jumps around the mod-90 range from frame to frame (unstable);
# latching the J6-target on the first confident frame grabs a wrong yaw (up to 45°) and the grasp fails. A near-square is
# graspable up to ±tens of degrees even at J6=0, so "wrong yaw" is riskier than "no-roll" → accumulate seg yaw as a fresh-frame EMA and
# rotate J6 only when temporally stable (EMA vector magnitude=concentration ≥ YAW_CONC_MIN); if instability persists, grasp with J6=0 (no-roll).
YAW_CONC_MIN = 0.85       # lower bound on seg-yaw EMA temporal concentration (0..1, 1=fully stable) — J6 rotation trusted only above this
YAW_ACC_BLEND = 0.2       # align seg-yaw EMA blend (per fresh perception frame; lower = longer averaging window)
# ⭐ The accumulation window must be long enough to "see" the swing (docs/33): MIN_N=6 (~0.2s) commits to an interval where a near-aligned seg is
# briefly stable at a wrong value, causing a 45° mis-rotation. The seg of a truly rotated cube (45° diamond) stays stable long, so
# it still passes with a much larger window, while near-aligned jumping clearly drops conc within the long window → J6=0 fallback.
# Accumulation happens in a fixed frame where J6 is not yet rotating (qf[5]=current while j6_target=None), so it is valid. Slow alignment is OK.
YAW_ACC_MIN_N = 45        # min fresh frames before the stability decision (~1.5s@30Hz) — long enough to clearly see the swing
YAW_ACC_GIVEUP_N = 80     # still unstable by this frame → J6=0 (no-roll) fallback (~2.7s@30Hz; just descend instead of abort)

# ---- FK-touch verification ----
FK_TOUCH_ARM_DZ = 0.025      # FK-touch: descend this much first before arming contact detection (skips hover false-fires from the descent-start acceleration transient)
#                              + re-baseline with the steady-descent torque at that point

# ---- Lift ----
LIFT_Z_MAX = 0.42            # 'Lift +z' safe TCP height upper bound (m, world) — so one click cannot push out of the workspace

# ---- Follow AUTO-CENTER (closed loop) — HW-measured decoupled wrist Jacobian (2026-07-20 controlled sweep) ----
# +2cm cam-fwd → wrist vn -0.38 (dvn/dfwd=-19/m, un unchanged); +2cm cam-lat → un -0.19 (dun/dlat=-9.5/m, vn unchanged).
# The measured signs make auto-center a stable Newton step (past runaways were caused by a WRONG-sign geometric model).
FOLLOW_DVN_DFWD = -19.0      # ∂(wrist vn)/∂(cam-fwd offset)  [1/m]
FOLLOW_DUN_DLAT = -9.5       # ∂(wrist un)/∂(cam-lat offset)  [1/m]
