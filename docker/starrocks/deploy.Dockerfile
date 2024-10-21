# This docker file build the Starrocks allin1 ubuntu image
# Please run this command from the git repo root directory to build:
#
#   - Use locally build artifacts to package runtime container:
#     > DOCKER_BUILDKIT=1 docker build --build-arg -f docker/starrocks/deploy.Dockerfile -t allin1-ubuntu:3.1.11 .
#
# The artifact source used for packing the runtime docker image
#   image: copy the artifacts from a artifact docker image.
#   local: copy the artifacts from a local repo. Mainly used for local development and test.

# create a docker build stage that copy locally build artifacts
FROM artifacts-ubuntu:3.1.11 as artifacts-from-local


FROM artifacts-from-local as artifacts
RUN rm -f /release/be_artifacts/be/lib/starrocks_be.debuginfo


FROM ubuntu:22.04 as dependencies-installed
ARG DEPLOYDIR=/data/deploy
ENV SR_HOME=${DEPLOYDIR}/starrocks
COPY docker/starrocks/conf/sources.list /etc/apt/sources.list

RUN apt-get update -y && apt-get install -y --no-install-recommends \
        binutils-dev default-jdk python2 mysql-client curl vim tree net-tools less tzdata linux-tools-common linux-tools-generic supervisor nginx netcat locales && \
        ln -fs /usr/share/zoneinfo/UTC /etc/localtime && \
        dpkg-reconfigure -f noninteractive tzdata && \
        locale-gen en_US.UTF-8 && \
        rm -rf /var/lib/apt/lists/*
RUN echo "export PATH=/usr/lib/linux-tools/5.15.0-60-generic:$PATH" >> /etc/bash.bashrc
ENV JAVA_HOME=/lib/jvm/default-java

WORKDIR $DEPLOYDIR

# Copy all artifacts to the runtime container image
COPY --from=artifacts /release/be_artifacts/ $DEPLOYDIR/starrocks
COPY --from=artifacts /release/fe_artifacts/ $DEPLOYDIR/starrocks
COPY --from=artifacts /release/broker_artifacts/ $DEPLOYDIR/starrocks

# Copy setup script and config files
COPY docker/starrocks/scripts/* $DEPLOYDIR
COPY docker/starrocks/services/ $SR_HOME

RUN cat be.conf >> $DEPLOYDIR/starrocks/be/conf/be.conf && \
    cat fe.conf >> $DEPLOYDIR/starrocks/fe/conf/fe.conf && \
    rm -f be.conf fe.conf && \
    mkdir -p $DEPLOYDIR/starrocks/fe/meta $DEPLOYDIR/starrocks/be/storage && touch /.dockerenv

CMD ./entrypoint.sh
