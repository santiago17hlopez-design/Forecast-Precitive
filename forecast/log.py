"""Console output with three verbosity levels.

Milestones always print. Detail prints only when asked. The distinction matters:
a scheduled run that floods the terminal gets ignored, and then the one line
that actually mattered — the data alert — scrolls past unread.
"""
from __future__ import annotations


class Log:
    def __init__(self, verbosity: str = "full"):
        # "quiet"    nothing but alerts
        # "moderate" milestones
        # "full"     everything
        self.verbosity = verbosity

    def detail(self, *args, **kwargs):
        if self.verbosity == "full":
            print(*args, **kwargs)

    def milestone(self, *args, **kwargs):
        if self.verbosity in ("full", "moderate"):
            print(*args, **kwargs)

    def alert(self, *args, **kwargs):
        """Always prints, whatever the verbosity."""
        print(*args, **kwargs)

    def title(self, text, width=78, char="="):
        self.detail("")
        self.detail(char * width)
        self.detail(f"  {text}")
        self.detail(char * width)

    def sub(self, text, width=78):
        self.detail("")
        self.detail(f"  {text}")
        self.detail("  " + "-" * (width - 2))
