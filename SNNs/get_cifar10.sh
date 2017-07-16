if [ ! -d "./cifar10_data" ]; then
  mkdir cifar10_data
  (cd cifar10_data; wget http://data.mxnet.io/mxnet/data/cifar10.zip)
  (cd cifar10_data; unzip -u *.zip)
fi
echo "Data downloaded"
