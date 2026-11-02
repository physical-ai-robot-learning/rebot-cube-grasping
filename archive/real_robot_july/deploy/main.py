#!/usr/bin/env python
"""reBot real-robot deployment launcher (entry point).

Puts src on the import path and calls the package's main(). All actual deployment logic
lives in src/rebot_sim2real/deploy/main.py (item 7: structured code goes under src/).
Run:  .venv-deploy/bin/python deploy/main.py [options…]   (the dashboard starts embedded alongside)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from rebot_sim2real.deploy.main import main   # noqa: E402 (import after path setup)

if __name__ == "__main__":
    main()
