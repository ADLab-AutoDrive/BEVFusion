if [ $# != 1 ]; then
    echo "Usage: bash ./docker/build_matrix_docker.sh code_version"
    exit
fi
TAG=$1
docker build . -f docker/Dockerfile_matrix_lite -t reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:autodrive3d_matrix_${TAG}
docker push reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:autodrive3d_matrix_${TAG}