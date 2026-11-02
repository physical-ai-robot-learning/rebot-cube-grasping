"""Live training visualization (sim render frames + SAC metrics) over a tiny HTTP server.

No tensorboard, no extra deps: stdlib ``http.server`` + Pillow(JPEG) + numpy.
See :mod:`rebot_sim2real.viz.dashboard`.
"""
