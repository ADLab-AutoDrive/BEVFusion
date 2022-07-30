if [ $# != 1 ]; then
    echo "Usage: bash ./docker/build_image.sh new_img_tag"
    exit
fi

NEW_IMG_TAG=$1

cat > ./Dockerfile.AutoML <<- EOM
FROM reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:mmdet3d
RUN mv /mnt /mnt_bk && ln -s /data /mnt
RUN mkdir -p /workspace
RUN /opt/conda/bin/pip install einops
ADD tools/autodrive_train.sh /bin/train
ADD tools/autodrive_eval.sh /bin/evaluate
RUN chmod +x /bin/train
RUN chmod +x /bin/evaluate
RUN touch /bin/inference && chmod +x /bin/inference
WORKDIR /workspace
EOM

docker build . -f ./Dockerfile.AutoML -t reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:mmdet3d_${NEW_IMG_TAG}
docker push reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:mmdet3d_${NEW_IMG_TAG}
rm ./Dockerfile.AutoML
docker tag reg.docker.alibaba-inc.com/had-perc/op-shuofan_test:mmdet3d_${NEW_IMG_TAG} reg.docker.alibaba-inc.com/had-perc/eval-shuofan_test:mmdet3d_${NEW_IMG_TAG}
docker push reg.docker.alibaba-inc.com/had-perc/eval-shuofan_test:mmdet3d_${NEW_IMG_TAG}

