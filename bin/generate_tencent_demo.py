import json

with open("examples/header_tencent.ipynb", "r") as f:
    header_tencent = json.load(f)

with open("examples/demo.ipynb", "r") as f:
    demo = json.load(f)

cells_to_insert = header_tencent["cells"]

# 查找文件B中包含 "# opensource or in tencent" 的 cell 的索引
index_to_insert = -1
for i, cell in enumerate(demo["cells"]):
    if any("# opensource or in tencent" in line for line in cell["source"]):
        index_to_insert = i
        break

# 如果找到了包含 "# opensource or in tencent" 的 cell，删除它并插入新的 cells
if index_to_insert != -1:
    del demo["cells"][index_to_insert]
    demo["cells"][index_to_insert:index_to_insert] = cells_to_insert


with open("examples/demo_tencent.ipynb", "w") as f:
    json.dump(demo, f, indent=2)