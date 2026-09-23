"""Build Spark's small reader against the installed SDK; stock stays intact."""
from pathlib import Path
import subprocess
import sys
import sysconfig

root = Path(__file__).resolve().parents[1]
destination = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/spark/app")
destination.mkdir(parents=True, exist_ok=True)
target = destination / ("spark_tof_native" + sysconfig.get_config_var("EXT_SUFFIX"))
temporary = target.with_suffix(".new")
subprocess.run(["g++", "-O2", "-shared", "-std=c++20", "-fPIC",
                "-I"+sysconfig.get_path("include"), "-I/.doly/libs/sdk/include",
                str(root/"native/tof_reader.cpp"), "-L/.doly/libs/sdk/lib",
                "-Wl,-rpath,/.doly/libs/sdk/lib", "-lTofControl", "-o", str(temporary)], check=True)
temporary.replace(target)
print(target)
