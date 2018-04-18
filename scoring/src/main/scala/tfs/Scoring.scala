package tfs

import breeze.linalg.DenseVector


object Scoring {

  def main(args: Array[String]): Unit = {
    val (m,g) = tfs.load("""C:\app\tf_export\tfs_model_214479""", "serve")

    val batchSize = 16
    val inputShape = Array[Long](batchSize, 8, 500, 20, 1)
    val len = inputShape.foldLeft(1L)(_ * _).toInt

    def inputData = DenseVector.rand[Double](len).data.map(_.toFloat)

    val s = new Score1(m.session())
    val a = s.predict(inputData)

    System.out.println(s"values:\n${a.take(1).map(_.mkString(", ")).mkString("\n")}")
  }
}
