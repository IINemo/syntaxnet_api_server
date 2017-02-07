# -*- coding: utf8 -*-

import sys

import os
import os.path

import SocketServer

################################################################################

def CreatePythonPathEntries(python_imports, module_space):
  parts = python_imports.split(':');
  return [module_space] + ["%s/%s" % (module_space, path) for path in parts]


module_space = '/root/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/'
python_imports = 'protobuf/python'
python_path_entries = CreatePythonPathEntries(python_imports, module_space)

repo_dirs = [os.path.join(module_space, d) for d in os.listdir(module_space)]
repositories = [d for d in repo_dirs if os.path.isdir(d)]
python_path_entries += repositories


sys.path += python_path_entries

################################################################################

import logging


def configure_logger():
    core_logger = logging.getLogger('common_logger')
    core_logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - '
                                  '%(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    core_logger.addHandler(ch)
    return core_logger


logger = configure_logger()

################################################################################

import shutil

custom_file_path = '/root/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/__main__/syntaxnet/api/context.pbtxt'
orig_file_path = '/root/models/syntaxnet/syntaxnet/models/parsey_universal/context.pbtxt'
shutil.copyfile(custom_file_path, orig_file_path)

import time
import tempfile
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

from google.protobuf import text_format

from syntaxnet import sentence_pb2
from syntaxnet import graph_builder
from syntaxnet import structured_graph_builder
from syntaxnet.ops import gen_parser_ops
from syntaxnet import task_spec_pb2

import StringIO
import select


beam_size = 8
max_steps = 1000
arg_prefix = 'brain_morpher'
slim_model = True
task_context_file = '/root/models/syntaxnet/syntaxnet/models/parsey_universal/context.pbtxt'
resource_dir = '/root/models/syntaxnet/syntaxnet/models/Russian-SynTagRus'
model_path = os.path.join(resource_dir, 'morpher-params')

custom_file_path = '/root/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/__main__/syntaxnet/api/1.txt'

input_str = 'custom_file'
#input_str = 'stdin-conll'
#input_str = 'stdin'

output_str = 'stdout-conll'
batch_size = 1024
hidden_layer_str = '64'

def RewriteContext(task_context_file):
  context = task_spec_pb2.TaskSpec()
  with gfile.FastGFile(task_context_file) as fin:
    text_format.Merge(fin.read(), context)

  for resource in context.input:
    for part in resource.part:
      if part.file_pattern != '-':
        part.file_pattern = os.path.join(resource_dir, part.file_pattern)

  with tempfile.NamedTemporaryFile(delete=False) as fout:
    fout.write(str(context))
    return fout.name


def init(sess, task_context):
  hidden_layer_sizes = map(int, hidden_layer_str.split(','))

  feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
    gen_parser_ops.feature_size(task_context=task_context,
      arg_prefix=arg_prefix))

  parser = structured_graph_builder.StructuredGraphBuilder(
          num_actions,
          feature_sizes,
          domain_sizes,
          embedding_dims,
          hidden_layer_sizes,
          gate_gradients=True,
          arg_prefix=arg_prefix,
          beam_size=beam_size,
          max_steps=max_steps)

  parser.AddEvaluation(task_context,
                       batch_size,
                       corpus_name=input_str,
                       evaluation_max_steps=max_steps)
  
  parser.AddSaver(slim_model)
  sess.run(parser.inits.values())
  parser.saver.restore(sess, model_path)
  
  return parser
  




class ProcessorSyntaxNet(object):
  def __init__(self):
    super(ProcessorSyntaxNet, self).__init__()
    hidden_layer_sizes = map(int, hidden_layer_str.split(','))

    feature_sizes, domain_sizes, embedding_dims, num_actions = sess.run(
      gen_parser_ops.feature_size(task_context=task_context,
        arg_prefix=arg_prefix))

    parser = structured_graph_builder.StructuredGraphBuilder(
            num_actions,
            feature_sizes,
            domain_sizes,
            embedding_dims,
            hidden_layer_sizes,
            gate_gradients=True,
            arg_prefix=arg_prefix,
            beam_size=beam_size,
            max_steps=max_steps)

    parser.AddEvaluation(task_context,
                         batch_size,
                         corpus_name=input_str,
                         evaluation_max_steps=max_steps)
    
    parser.AddSaver(slim_model)
    sess.run(parser.inits.values())
    parser.saver.restore(sess, model_path)
    
    return parser

  def parse(self, raw_bytes):
    with open(custom_file_path, 'w') as f:
      pass 

    self._parse_impl(self.server.parser_, self.server.task_context_)
    sys.stdout.flush()
    
    with open(custom_file_path, 'a') as f:
      f.write('\n' + data)
      f.flush()

    self._parse_impl(self.server.parser_, self.server.task_context_)
    sys.stdout.flush()

  def _parse_iml(self, parser, task_context):
    tf_eval_epochs, tf_eval_metrics, tf_documents = sess.run([
          parser.evaluation['epochs'],
          parser.evaluation['eval_metrics'],
          parser.evaluation['documents']
     ])

    sink_documents = tf.placeholder(tf.string)
    sink = gen_parser_ops.document_sink(sink_documents,
                                        task_context=task_context,
                                        corpus_name=output_str)

    sess.run(sink, feed_dict={sink_documents: tf_documents})


class SyncHandler(SocketServer.BaseRequestHandler):
  def read_incoming_request(self):
    chunk = self.request.recv(1024)
    data = str()
    data += chunk
    while '\n' not in chunk:
      chunk = self.request.recv(1024)
      data += chunk

    return data

  def handle(self):
    logger.debug('Incoming request.')

    data = self.read_incoming_request()

    with open(custom_file_path, 'w') as f:
      pass 

    parse(self.server.parser_, self.server.task_context_)
    
    with open(custom_file_path, 'a') as f:
      f.write('\n' + data)
      f.flush()

    sys.stdout.flush()

    parse(self.server.parser_, self.server.task_context_)
    sys.stdout.flush()

    logger.debug('Parsing finished.')

    reply = self.read_all_stream(self.server.read_stream_)
    self.request.sendall(reply)

  def read_all_stream(self, strm):
    # pollster = select.poll()
    # flags = select.POLLIN
    # pollster.register(strm.fileno(), flags)

    result = str()
    # line = strm.read(100)
    # result = line
    # try:
    #   while pollster.poll(0)[1]:
    #     line = strm.read(1024)
    #     if line:
    #       result += line
    #     else:
    #       break
    # except:
    #   pass

    max_read = 1024
    while True:
      chunk = os.read(strm.fileno(), max_read)
      result += chunk
      if len(chunk) < max_read:
        break

    return result


def configure_stdout():
  read_stream, write_stream = os.pipe()
  os.dup2(write_stream, sys.stdout.fileno())
  return os.fdopen(read_stream)


def main(sess):
  sync_server = SocketServer.TCPServer(('0.0.0.0', 9999), SyncHandler)
  sync_server.task_context_ = RewriteContext(task_context_file)
  sync_server.parser_ = init(sess, sync_server.task_context_)
  sync_server.read_stream_ = configure_stdout()
  sync_server.serve_forever()  


if __name__ == '__main__':
  with tf.Session() as sess:
    main(sess)
