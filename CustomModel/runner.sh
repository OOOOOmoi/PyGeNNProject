# find /home/yangjinhao/PyGenn/CustomModel/output -name "*.png" -type f -delete
python CustomModel.py --duration 1000 --device 2 --poisson --buffer --buffer-size 100 --AreaIdx 0 --stim --stim-start 400 --stim-end 800
