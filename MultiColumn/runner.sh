find /home/yangjinhao/PyGenn/MultiColumn -name "*.png" -type f -delete
python MultiColumn.py --duration 1500 --stim 50 --stimStart 600 --stimEnd 1200
cd output
python plot.py --drop 200 --layer-psd --pop-psd --area-psd
cd ..