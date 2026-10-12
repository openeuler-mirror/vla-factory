"""Loss-parity capture scripts.

These files are executed by absolute path inside candidate venvs
(``<judge-python> .../factory_parity_runner.py <subcommand>``), not imported
into the harness process, so they stay self-contained: ``parity_io`` is
imported through the script directory landing on ``sys.path``. Do not add
package-relative imports here.
"""

from __future__ import annotations
