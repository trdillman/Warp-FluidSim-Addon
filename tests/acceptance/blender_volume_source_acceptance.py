import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from _phase2_source_acceptance_common import main, setup
setup()
main("volume_source")
