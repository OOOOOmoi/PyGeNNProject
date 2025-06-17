# find /home/yangjinhao/PyGenn/MultiColumn -name "*.png" -type f -delete
python MultiColumn.py --duration 3000 --stim 0 --stimStart 600 --stimEnd 1200
cd output
python plot.py --drop 200
cd ..