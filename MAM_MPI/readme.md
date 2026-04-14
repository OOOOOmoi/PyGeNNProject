该项目为GeNN+MPI模拟猕猴大脑视觉皮层模型的源代码
依赖软件包为pygenn、numba，pygenn的安装见https://github.com/genn-team/genn
主体代码为CustomModel_MPI.py，将各项输入输出路径设置为自己的路径之后直接运行即可

生成代码存储位置:GenCODE/...
运行时间详细记录:log/...
可视化结果:output/...

config.py: 模型设置，包括一些基本的参数信息
connectom.py: 绘制连接矩阵所用代码
getStruct.py: 获取模型结构信息（神经元、突触数量、突触权重、突出延迟等）
inSyn.py: 记录模型电流所用函数，目前没用到
psd.py: 绘制群体、层、区域不同分辨率下功率谱密度函数
rate_curve.py: 获得群体、层、区域不同分辨率下放电率曲线的函数
read_test.py: 一些小测试demo在此完成
record.py: 记录、保存spike所用函数
update_time.py: 可视化通信用时所用函数
visual.py: 可视化主体代码，主要包括raster、hist、电流、放电率、功率谱等