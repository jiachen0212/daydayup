schedctl create --name build  --image "harbor.smoa.cc/public/smore_core:v2.2.0.cu10" --gpu 1 --cmd "cd /newdata/jiachen/project/det/FCOS && /newdata/jiachen/miniconda3/bin/python setup.py build develop --no-deps"

# pytorch: 1.4.0 