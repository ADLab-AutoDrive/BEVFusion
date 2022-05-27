if [ $# != 2 ]; then
    echo "Usage: bash ./docker/build_base_docker.sh code_version"
    exit
fi
TAG=$1
GPU=$2
if [[ $GPU == "3090" ]]
then
  echo "#### GPU: 3090####"
  docker build . -f docker/3090.Dockerfile -t reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:autodrive3d_base_${TAG}
else
  docker build . -f docker/Dockerfile -t reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:autodrive3d_base_${TAG}
fi
docker push reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:autodrive3d_base_${TAG}
docker tag  reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:autodrive3d_base_${TAG} reg.docker.alibaba-inc.com/had-perc/op-mxlidar_ssl:autodrive3d_base_${TAG}
docker push reg.docker.alibaba-inc.com/had-perc/op-mxlidar_ssl:autodrive3d_base_${TAG}