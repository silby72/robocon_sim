"""GUI IO: comment-preserving YAML load/save + display-unit <-> SI conversion.

This is the ONLY place mm/degree display units are allowed to exist; everything
handed to the YAML document is SI (metres / radians). The core never sees
display units.
"""
