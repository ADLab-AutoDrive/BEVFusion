if [ $# != 1 ]; then
    echo "Usage: bash ./docker/build_data_process_docker.sh version"
    exit
fi
TAG=$1
docker build . -f docker/data_process.Dockerfile -t reg.docker.alibaba-inc.com/had-perc/process-mxlidar:a3l_${TAG}
docker push reg.docker.alibaba-inc.com/had-perc/process-mxlidar:a3l_${TAG}