package tfs

import java.io.{File, FileOutputStream, OutputStreamWriter, Writer}
import java.nio.charset.StandardCharsets

import breeze.linalg.DenseVector
import org.tensorflow.Tensor

import scala.collection.JavaConverters._

object Example {

  def getWriter(name: String): Writer = {
    val os = new FileOutputStream(new File(s"C:\\app\\tmp\\$name.log"))
    new OutputStreamWriter(os, StandardCharsets.UTF_8)
  }

  def write(name: String, s: String): Unit = {
    val ws = getWriter(name)
    ws.write(s)
    ws.flush()
    ws.close()
  }

  def main(args: Array[String]): Unit = {
    val (m,g) = tfs.load("""C:\app\tf_export\tfs_model_214479""", "serve")
    write("Graph", m.graph.prettyPrint)
    write("Operations", m.graph().operations().asScala.map(_.name()).mkString("\n"))
    write("MetaGraphDef", g.toString)

    val w = getWriter("SignatureDefs")
    val s = g.sigs()
    s.foreach{x =>
      w.write(s"Signature\n")
      w.write(s"name: ${x._1}\n")
      w.write(s"method: ${x._2.getMethodName}\n")
      w.write("inputs: \n")
      x._2.inputs.foreach(x1 => w.write("input: " + x1._1 + "\n" + x1._2.prettyPrint + "\n\n"))
      w.write("outputs: \n")
      x._2.outputs.foreach(x1 => w.write("output: " + x1._1 + ": " + x1._2.prettyPrint + "\n\n"))
    }
    w.flush()
    w.close()

    val batchSize = 16
    val inputShape = Array[Long](batchSize, 8, 500, 20, 1)
    val len = inputShape.foldLeft(1L)(_ * _).toInt

    val inputData = DenseVector.rand[Double](len).data.map(_.toFloat)
    val tensor = tfs.fromVector(data = inputData, shape = inputShape)
    write("Tensor", tensor.toString)

    val sess = m.session()
    // see MetaGraphDef / meta_info_def / stripped_op_list / op / input_arg

    val out: Seq[Tensor[_]] = sess.runner()
      .feed("memory/images", tensor)
      .fetch("output/scores/BiasAdd")
      .run()
      .asScala

    val a = out.head.copyTo(Array.fill[Float](batchSize, 4){0})
    write("BiasAdd", s"tensor:\n${out.toString}\n\nvalues:\n${a.map(_.mkString(", ")).mkString("\n")}")
  }
}
