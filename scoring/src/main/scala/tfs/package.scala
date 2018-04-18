import org.tensorflow._
import org.tensorflow.framework.{MetaGraphDef, SignatureDef, TensorInfo}

import scala.collection.JavaConverters._
import java.nio.FloatBuffer
import java.nio.file.{Files, Paths}

import scala.util.Try

package object tfs {
  def graphFromPB(pbPath: String): Graph = {
    val g = new Graph()
    g.importGraphDef(Files.readAllBytes(Paths.get(pbPath)))
    g
  }

  def sessionFromPB(pbPath: String): Session = new Session(graphFromPB(pbPath))

  /**
    * TensorFlow saved model representation.
    *
    * @param exportDir A source the model is loaded from.
    * @param tags tags which uniquely identifies a computational graph defined in the saved model.
    * @return MetaGraphDef TensorFlow API object representing a computational graph.
    */
  def load(exportDir: String, tags: String*): (SavedModelBundle, MetaGraphDef) = {
    val savedModelBundle = SavedModelBundle.load(exportDir, tags:_*)
    (savedModelBundle, savedModelBundle.graphDef())
  }

  implicit class DataType1 (x: DataType) {
    def prettyPrint: String = "DataType: " + Try(x.name()).getOrElse(x.ordinal())
  }

  implicit class Shape1(x: Shape) {
    def prettyPrint: String = s"Shape: ${x.toString}"
  }

  implicit class Output1(x: Output[_]) {
    def prettyPrint: String = {
      s"Output ${x.index()}: ${x.op().name()} ${Try(x.dataType().prettyPrint).getOrElse(x.index().toString)} ${x.shape().prettyPrint}"
    }
  }

  implicit class Operation1(x: Operation) {
    def prettyPrint: String = {
      s"${x.name()}: ${x.`type`()} with ${x.numOutputs()} outputs {\n" +
      (0 until x.numOutputs()).map(x.output).map(o => s"\t\t${o.prettyPrint}").mkString("\n") +
      "\n}\n"
    }
  }

  implicit class Graph1(x: Graph) {
    def prettyPrint: String = {
      s"Graph:\n" +
      "\tOperations:\n" +
      x.operations().asScala.map(o => "\t\t" + o.prettyPrint).mkString("\n")
    }
  }

  implicit class SignatureDef1(x: SignatureDef) {
    def inputs: Seq[(String,TensorInfo)] = x.getInputsMap.asScala.toSeq
    def outputs: Seq[(String,TensorInfo)] = x.getOutputsMap.asScala.toSeq
    def input(name: String): TensorInfo = x.getInputsOrThrow(name)
    def output(name: String): TensorInfo = x.getOutputsOrThrow(name)
  }

  implicit class TensorInfo1(x: TensorInfo) {
    def opName: String = x.getName.substring(0, x.getName.indexOf(':'))
    def shape: Array[Long] = x.getTensorShape.getDimList.asScala.map(_.getSize).toArray
    def prettyPrint: String =
      s"""name: ${x.getName}
        |  opName: $opName
        |  shape: (${shape.mkString(", ")})
      """.stripMargin

    def opNameAndShape: (String, Array[Long]) = (opName, shape)
  }

  implicit class SavedModelBundle1(x: SavedModelBundle) {
    def graphDef(): MetaGraphDef = MetaGraphDef.parseFrom(x.metaGraphDef())
    def signatureDef(name: String): SignatureDef = x.graphDef().signatureDef(name)
    def input(sigName: String, tensorName: String): TensorInfo = signatureDef(sigName).input(tensorName)
    def output(sigName: String, tensorName: String): TensorInfo = signatureDef(sigName).output(tensorName)
  }

  implicit class MetaGraphDef1(x: MetaGraphDef) {
    def signatureDefMap: Map[String, SignatureDef] = x.getSignatureDefMap.asScala.toMap
    def sigs(): Seq[(String, SignatureDef)] = x.getSignatureDefMap.asScala.toSeq
    def signatureDef(name: String): SignatureDef = x.getSignatureDefOrThrow(name)
  }

  def fromVector(data: Array[Float], shape: Array[Long]): Tensor[_] = Tensor.create(shape, FloatBuffer.wrap(data))

  def fromMatrix(data: Array[Array[Float]], shape: Array[Long]): Tensor[_] = Tensor.create(shape, FloatBuffer.wrap(data.flatten))

  def toVector(tensor: Tensor[Float]): Array[Float] = {
    val array = Array.ofDim[Float](tensor.shape.head.toInt)
    tensor.copyTo(array)
    array
  }

  def toMatrix(tensor: Tensor[Float]): Array[Array[Float]] = {
    val array = Array.ofDim[Float](tensor.shape()(0).toInt, tensor.shape()(1).toInt)
    tensor.copyTo(array)
    array
  }

  /** prediction from Array
    *
    * @param session
    * @param inputData
    * @param buf
    * @param inputShape
    * @param outputShape
    * @param inputOp
    * @param outputOp
    * @return
    */
  def predict(session: Session,
              inputData: Array[Float],
              buf: Array[_],
              inputShape: Array[Long],
              outputShape: Array[Long],
              inputOp: String,
              outputOp: String): Array[_] =
    predict(session,
            tfs.fromVector(inputData, inputShape),
            buf,
            inputShape: Array[Long],
            outputShape: Array[Long],
            inputOp: String,
            outputOp: String)

  /** Prediction from Tensor
    *
    * @param session
    * @param inputTensor
    * @param buf
    * @param inputShape
    * @param outputShape
    * @param inputOp
    * @param outputOp
    * @return
    */
  def predict(session: Session,
              inputTensor: Tensor[_],
              buf: Array[_],
              inputShape: Array[Long],
              outputShape: Array[Long],
              inputOp: String,
              outputOp: String): Array[_] = {
    val outputTensors: Seq[Tensor[_]] = session.runner()
      .feed(inputOp, inputTensor)
      .fetch(outputOp)
      .run()
      .asScala
    outputTensors.head.copyTo(buf) // copy first tensor
    outputTensors.foreach(_.close()) // free memory allocated by libtensorflow jni
    buf
  }
}
