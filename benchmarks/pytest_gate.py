"""Pytest plugin recording executed tests for the benchmark correctness gate."""

from .common import write_report


def pytest_addoption(parser):
    parser.addoption("--benchmark-gate-report", help="Write benchmark correctness counts as JSON")
    parser.addoption("--benchmark-required-marker", default="cuda")


def pytest_configure(config):
    path = config.getoption("--benchmark-gate-report")
    if path:
        config.pluginmanager.register(GateReport(path, config.getoption("--benchmark-required-marker")))


class GateReport:
    def __init__(self, path, marker):
        self.path, self.marker = path, marker
        self.required = set()
        self.passed = set()
        self.failed = set()
        self.skipped = set()
        self.collected = 0

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)
        self.required = {item.nodeid for item in session.items if item.get_closest_marker(self.marker)}

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed.add(report.nodeid)
        elif report.skipped:
            self.skipped.add(report.nodeid)
        elif report.when == "call" and report.passed and not hasattr(report, "wasxfail"):
            self.passed.add(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        passed = self.passed - self.failed - self.skipped
        required_passed = len(passed & self.required)
        write_report(
            self.path,
            {
                "status": "passed" if exitstatus == 0 and required_passed else "failed",
                "exit_code": int(exitstatus),
                "collected": self.collected,
                "passed": len(passed),
                "failed": len(self.failed),
                "skipped": len(self.skipped),
                "required_marker": self.marker,
                "required_passed": required_passed,
            },
        )
