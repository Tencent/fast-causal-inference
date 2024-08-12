# This docker file build the Starrocks artifacts fe & be and package them into a busybox-based image
# Please run this command from the git repo root directory to build:
#
# Build a Ubuntu based artifact image:
#  > DOCKER_BUILDKIT=1 docker build --rm=true --build-arg builder=starrocks/dev-env-ubuntu:3.1.1 -f docker/starrocks/dev.Dockerfile -t artifacts-ubuntu:3.1.1 .

FROM starrocks/dev-env-ubuntu:3.1.1 as fe-builder
ARG BUILD_TYPE=Release
ARG MAVEN_OPTS="-Dmaven.artifact.threads=8"


COPY ./contrib/starrocks /build/starrocks
WORKDIR /build/starrocks
# clean and build Frontend and Spark Dpp application
RUN  BUILD_TYPE=${BUILD_TYPE} MAVEN_OPTS=${MAVEN_OPTS} ./build.sh --fe --clean


FROM starrocks/dev-env-ubuntu:3.1.1 as broker-builder
ARG MAVEN_OPTS
COPY ./contrib/starrocks /build/starrocks
WORKDIR /build/starrocks
# clean and build Frontend and Spark Dpp application
RUN  cd fs_brokers/apache_hdfs_broker/ && MAVEN_OPTS=${MAVEN_OPTS} ./build.sh


FROM starrocks/dev-env-ubuntu:3.1.1 as be-builder
ARG MAVEN_OPTS
# build Backend in different mode (build_type could be Release, DEBUG, or ASAN). Default value is Release.
ARG BUILD_TYPE
COPY ./contrib/starrocks /build/starrocks
WORKDIR /build/starrocks
RUN  BUILD_TYPE=${BUILD_TYPE} MAVEN_OPTS=${MAVEN_OPTS} ./build.sh --be --clean -j6

FROM busybox:latest

LABEL org.opencontainers.image.source="https://github.com/Tencent/fast-causal-inference"
LABEL org.starrocks.version=${RELEASE_VERSION:-"FastCausalInference"}

COPY --from=fe-builder /build/starrocks/output /release/fe_artifacts
COPY --from=be-builder /build/starrocks/output /release/be_artifacts
COPY --from=broker-builder /build/starrocks/fs_brokers/apache_hdfs_broker/output /release/broker_artifacts

WORKDIR /release
