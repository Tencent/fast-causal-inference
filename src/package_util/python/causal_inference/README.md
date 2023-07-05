causal inference package.
contain ols, ttest, delta-method etc...

tencent inner dev env pypi:
pip install pyspark==3.1.2
pip install tdw-pytoolkit -i https://bearlyhuang:xxxx@mirrors.tencent.com/repository/pypi/tencent_pypi/simple

install:
yum install -y libjpeg-devel
yum install python3-devel -y
yum install python-devel -y

mkdir ~/.pip
echo "[global]" >~/.pip/pip.conf
echo "index-url= https://mirrors.tencent.com/pypi/simple/" >> ~/.pip/pip.conf

pip3.6 install requests --trusted-host mirrors.tencent.com
