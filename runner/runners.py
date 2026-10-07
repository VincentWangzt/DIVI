from runner.sivi import SIVIRunner
from runner.uivi import UIVIRunner
from runner.aisivi import AISIVIRunner
from runner.divi import DIVIRunner
from runner.ksivi import KSIVIRunner
from runner.kpg import KPGRunner
from runner.nfvi import NFVIRunner
from runner.base_runner import BaseSIVIRunner

Runners: dict[str, type[BaseSIVIRunner]] = {
    "SIVI": SIVIRunner,
    "UIVI": UIVIRunner,
    "AISIVI": AISIVIRunner,
    "DIVI": DIVIRunner,
    "KSIVI": KSIVIRunner,
    "KPG": KPGRunner,
    "NFVI": NFVIRunner,
}
