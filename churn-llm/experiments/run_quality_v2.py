"""최신 품질 개선 비교 실험 CLI."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from runtime.quality_v2 import main

if __name__=='__main__': main()
