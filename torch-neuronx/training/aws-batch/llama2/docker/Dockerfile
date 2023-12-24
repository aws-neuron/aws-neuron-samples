ARG BASE_IMAGE_REPO
ARG BASE_IMAGE_NAME
ARG BASE_IMAGE_TAG
ARG DEBIAN_FRONTEND=noninteractive

FROM ${BASE_IMAGE_REPO}/${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}

COPY ./llama2 /

WORKDIR /
RUN chmod +x /llama_batch_training.sh
CMD ["/llama_batch_training.sh"]