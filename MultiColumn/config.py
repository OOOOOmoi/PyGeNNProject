import json
import os
current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
DataPath=os.path.join(parent_dir, "custom_Data_Model_3396.json")
with open(DataPath, 'r') as f:
    ParamOfAll = json.load(f)
# AreaList=["V1"]
AreaList=[
        "V1",
        "V2",
        "VP",
        "V3",
        "V3A",
        "MT",
        "V4t",
        "V4"]
# AreaList=ParamOfAll['area_list']
# for i in range(10):
#     print(i)