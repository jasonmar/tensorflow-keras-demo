package tfs

import breeze.linalg.DenseVector

object ScoringWithModel {
  def main(args: Array[String]): Unit = {
    val model = Model("""C:\app\tf_export\tfs_model_214479""", arrayFn = () => Array.fill[Float](16, 4){0})
    val inputData: Array[Float] = DenseVector.rand[Double](model.inputParamCount).data.map(_.toFloat)
    model.predict(inputData) match {
      case a: Array[Array[Float]] =>
        System.out.println(a.map(_.mkString("[ ", ", ", " ]")).mkString("\n"))
      case a: Array[Float] =>
        System.out.println(a.mkString("[", ", ", "]"))
      case _ =>
    }
  }
}
