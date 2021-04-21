find src -name "*.java" -print | xargs javac -d dist
java -cp dist nnimpl.Project
# java -jar fi-muni-pv021-automatic-evaluator-1.0-SNAPSHOT-jar-with-dependencies.jar actualTestPredictions expectedTestPredictions  10
