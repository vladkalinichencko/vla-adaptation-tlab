from pathlib import Path

from convert_libero import convert


SOURCE = Path("/private/tmp/libero-inspect/libero_90/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5")
OUTPUT = Path("/private/tmp/official_libero_90_v3_test")


print(convert([SOURCE], OUTPUT))
