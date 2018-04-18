package tfs

import org.tensorflow.{Session, Tensor}
import scala.collection.JavaConverters._


class Score1(sess: Session){
  def predict(inputData: Array[Float], batchSize: Int = 16): Array[Array[Float]] = {
    val inputShape = Array[Long](batchSize, 8, 500, 20, 1)
    val tensor = tfs.fromVector(data = inputData, shape = inputShape)
    val out: Seq[Tensor[_]] = sess.runner()
      .feed("memory/images", tensor)
      .fetch("output/scores/BiasAdd")
      .run()
      .asScala

    out.head.copyTo(Array.fill[Float](batchSize, 4){0})
  }
}
