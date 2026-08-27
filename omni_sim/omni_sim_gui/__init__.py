"""omni_sim_gui -- config editor for the robot description.

Strict rule (spec section 1): this GUI is an *editor for YAML files* and nothing
more. Its only output is YAML under ``config/robot/``. Dependency direction is
one-way, ``gui -> core``: it may import ``omni_sim_core`` (schema, jacobian,
comment-preserving YAML io), but ``omni_sim_core`` must never import this
package. The simulator does not know the GUI exists.

Qt binding: PySide6 is the target (spec section 6.1). Until it is installed this
package runs on PyQt5 via a thin shim (``omni_sim_gui.qt``); the application code
is binding-agnostic, so switching to PySide6 later needs no code changes.
"""
