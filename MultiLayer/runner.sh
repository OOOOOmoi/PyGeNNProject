# find /home/yangjinhao/PyGenn/MultiLayer -name "*.png" -type f -delete
python MultiLayer.py --duration 1000 --stim 0 --stimStart 600 --stimEnd 1200
cd output
python plot.py --drop 200
cd ..