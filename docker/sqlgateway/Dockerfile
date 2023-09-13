FROM java:openjdk-8u111
LABEL maintainer="bearlyhuang@tencent.com"
WORKDIR /work
EXPOSE 9099
COPY lib/sqlgateway-0.0.1-SNAPSHOT.jar sqlgateway.jar
ENV JAVA_OPTS="-Dspring.config.location=/work/application.yml"
ENTRYPOINT exec java $JAVA_OPTS -jar sqlgateway.jar
