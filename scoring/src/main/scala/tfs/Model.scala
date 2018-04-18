package tfs

import org.tensorflow.{SavedModelBundle, Session}
import tfs.Model.Prediction

object Model {
  def apply(exportDir: String,
            tag: String = "serve",
            sigName: String = "predict",
            inName: String = "images",
            outName: String = "scores",
            arrayFn: () => Array[_]): Model = {
    val model = SavedModelBundle.load(exportDir, tag)
    new Model(model, sigName, inName, outName, arrayFn)
  }

  trait HasSession {
    val session: Session
  }

  trait InputAndOutput {
    protected val inputOp: String
    protected val inputShape: Array[Long]
    protected val outputOp: String
    protected val outputShape: Array[Long]
    def getInputShape: Seq[Long] = inputShape.toSeq
    def getOutputShape: Seq[Long] = outputShape.toSeq
    def inputParamCount: Int = inputShape.foldLeft(1L)(_ * _).toInt
    def outputParamCount: Int = outputShape.foldLeft(1L)(_ * _).toInt
  }

  trait Prediction extends HasSession with InputAndOutput {
    def predict(inputData: Array[Float], buf: Array[_]): Array[_] =
      tfs.predict(session, inputData, buf, inputShape, outputShape, inputOp, outputOp)
  }

  class Model2(override val session: Session,
               override val inputOp: String,
               override val inputShape: Array[Long],
               override val outputOp: String,
               override val outputShape: Array[Long]) extends Prediction
}

class Model(model: SavedModelBundle,
            sigName: String,
            inName: String,
            outName: String,
            arrayFn: () => Array[_]) extends Prediction{
  override protected val session: Session = model.session()

  override protected val (inputOp: String, inputShape: Array[Long]) =
    model.input(sigName, inName).opNameAndShape

  override protected val (outputOp: String, outputShape: Array[Long]) =
    model.output(sigName, outName).opNameAndShape

  def predict(inputData: Array[Float]): Array[_] = predict(inputData, arrayFn())

  /** Free memory held by libtensorflow jni */
  def close(): Unit = {
    session.close()
    model.close()
  }
}
