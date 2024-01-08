##add something so I don't have to change the version number by hand!
rm dist/*
python3 setup.py sdist bdist_wheel
auditwheel repair dist/valeriepieris-0.1.4-cp38-cp38-linux_x86_64.whl --plat manylinux_2_17_x86_64
mv wheelhouse/* dist; rm dist/valeriepieris-0.1.4-cp38-cp38-linux_x86_64.whl
python3 -m twine upload --repository pypi dist/*
