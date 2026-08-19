"""Page modules for the Copyright Detective application.

Keep package initialization side-effect free. Page implementations are imported
directly by their callers so a deployment reload never loads unrelated, heavy
modules while Python is still initializing ``src.pages``.
"""