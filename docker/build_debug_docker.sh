if [ $# != 1 ]; then
    echo "Usage: bash ./docker/build_base_docker.sh code_version"
    exit
fi
TAG=$1
cat docker/Dockerfile_debug | sed "s/mmdet3d_new/autodrive3d_base_${TAG}/" > docker/Dockerfile_tmp
docker build . -f docker/Dockerfile_tmp -t reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:autodrive3d_debug_${TAG}
docker push reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:autodrive3d_debug_${TAG}
rm -f docker/Dockerfile_tmp