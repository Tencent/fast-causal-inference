sudo ../gradlew clean
sudo ../gradlew assemble
sudo chmod 777 build/libs/calcite-core-1.36.0-SNAPSHOT.jar
if [ -e ../../sqlgateway/src/main/resources/lib/calcite-core-1.36.0-SNAPSHOT.jar ]; then
    echo "The file exists."
    rm -f ../../sqlgateway/src/main/resources/lib/calcite-core-1.36.0-SNAPSHOT.jar
fi
sudo mv build/libs/calcite-core-1.36.0-SNAPSHOT.jar ../../sqlgateway/src/main/resources/lib/
