version=0.0.2
sed -e s/VERSION_NUMBER/$version/g pyproject.toml.template > pyproject.toml
sed -e s/VERSION_NUMBER/$version/g setup.cfg.template > setup.cfg
rm dist/*
python3 setup.py sdist bdist_wheel
auditwheel repair dist/spatialsignificance-"$version"-cp310-cp310-linux_x86_64.whl --plat manylinux_2_17_x86_64
mv wheelhouse/* dist; rm dist/spatialsignificance-"$version"-cp310-cp310-linux_x86_64.whl
python3 -m twine upload --repository pypi dist/* --verbose
