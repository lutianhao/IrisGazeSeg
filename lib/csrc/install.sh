cd dcn_v2
rm *.so
rm -r build/
python setup.py build_ext --inplace
cd ../extreme_utils
rm *.so
rm -r build/
python setup.py build_ext --inplace
cd ../roi_align_layer
rm *.so
rm -r build/
python setup.py build_ext --inplace