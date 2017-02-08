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

################################################################################

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


class ProcessorSyntaxNetConfig(object):
  def __init__(self, 
    beam_size, 
    max_steps, 
    arg_prefix, 
    slim_model, 
    task_context_file, 
    resource_dir,
    model_path, 
    batch_size, 
    hidden_layer_str,
    custom_file_path,
    input_str,
    variable_scope,
    init_line,
    flush_input = False,
    max_tmp_size = 524288000):

    self.beam_size = beam_size
    self.max_steps = max_steps
    self.arg_prefix = arg_prefix
    self.slim_model = slim_model
    self.task_context_file = task_context_file
    self.resource_dir = resource_dir
    self.model_path = model_path
    self.batch_size = batch_size
    self.hidden_layer_str = hidden_layer_str

    self.custom_file_path = custom_file_path
    self.input_str = input_str
    self.variable_scope = variable_scope
    self.flush_input = flush_input
    self.max_tmp_size = max_tmp_size
    self.init_line = init_line


task_context_file = '/root/models/syntaxnet/syntaxnet/models/parsey_universal/context.pbtxt'
resource_dir = '/root/models/syntaxnet/syntaxnet/models/Russian-SynTagRus'
custom_file_dir = '/dev/shm/'


CFG_MORPH_PARSER = ProcessorSyntaxNetConfig(
  beam_size = 8,
  max_steps = 1000,
  arg_prefix = 'brain_morpher',
  slim_model = True,
  task_context_file = task_context_file,
  resource_dir = resource_dir,
  model_path = os.path.join(resource_dir, 'morpher-params'),
  batch_size = 1024,
  hidden_layer_str = '64',

  custom_file_path = os.path.join(custom_file_dir, 'morpher.tmp'),
  input_str = 'custom_file_morpher',
  variable_scope = 'morpher',
  flush_input = True,
  max_tmp_size = 524288000,
  init_line = '1')


CFG_MORPH_TAGGER = ProcessorSyntaxNetConfig(
  beam_size = 8,
  max_steps = 1000,
  arg_prefix = 'brain_tagger',
  slim_model = True,
  task_context_file = task_context_file,
  resource_dir = resource_dir,
  model_path = os.path.join(resource_dir, 'tagger-params'),
  batch_size = 1024,
  hidden_layer_str = '64',

  custom_file_path = os.path.join(custom_file_dir, 'tagger.tmp'),
  input_str = 'custom_file_tagger',
  variable_scope = 'tagger',
  max_tmp_size = 524288000,
  init_line = '1\t'*10)


CFG_SYNTAX_PARSER = ProcessorSyntaxNetConfig(
  beam_size = 8,
  max_steps = 1000,
  arg_prefix = 'brain_parser',
  slim_model = True,
  task_context_file = task_context_file,
  resource_dir = resource_dir,
  model_path = os.path.join(resource_dir, 'parser-params'),
  batch_size = 1024,
  hidden_layer_str = '512,512',

  custom_file_path = os.path.join(custom_file_dir, 'parser.tmp'),
  input_str = 'custom_file_parser',
  variable_scope = 'synt_parser',
  max_tmp_size = 524288000,
  init_line = '1\t'*12)


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


class ProcessorSyntaxNet(object):
  def __init__(self, cfg, read_stream):
    super(ProcessorSyntaxNet, self).__init__()

    self.parser_ = None
    self.task_context_ = RewriteContext(task_context_file)
    self.read_stream_ = read_stream
    self.sess_ = tf.Session()
    self.cfg_ = cfg

    with open(self.cfg_.custom_file_path, 'w') as f:
      pass

    self.fdescr_ = open(self.cfg_.custom_file_path, 'r')

    hidden_layer_sizes = map(int, self.cfg_.hidden_layer_str.split(','))

    with tf.variable_scope(self.cfg_.variable_scope):
      feature_sizes, domain_sizes, embedding_dims, num_actions = self.sess_.run(
        gen_parser_ops.feature_size(task_context=self.task_context_,
          arg_prefix=self.cfg_.arg_prefix))

      self.parser_ = structured_graph_builder.StructuredGraphBuilder(
              num_actions,
              feature_sizes,
              domain_sizes,
              embedding_dims,
              hidden_layer_sizes,
              gate_gradients=True,
              arg_prefix=self.cfg_.arg_prefix,
              beam_size=self.cfg_.beam_size,
              max_steps=self.cfg_.max_steps)

      self.parser_.AddEvaluation(self.task_context_,
        self.cfg_.batch_size,
        corpus_name=self.cfg_.input_str,
        evaluation_max_steps=self.cfg_.max_steps)
      
      self.parser_.AddSaver(self.cfg_.slim_model)
      self.sess_.run(self.parser_.inits.values())
      self.parser_.saver.restore(self.sess_, self.cfg_.model_path)

      self.parse(self.cfg_.init_line)

  def parse(self, raw_bytes):
    if self.cfg_.flush_input and self.fdescr_.tell() > self.cfg_.max_tmp_size:
      with open(self.cfg_.custom_file_path, 'w') as f:
        pass 

      self._parse_impl()
    
    with open(self.cfg_.custom_file_path, 'a') as f:
      f.write(raw_bytes)
      f.flush()

    self._parse_impl()

    result = self._read_all_stream(self.read_stream_)
    return result

  def _parse_impl(self):
    with tf.variable_scope(self.cfg_.variable_scope):
      tf_eval_epochs, tf_eval_metrics, tf_documents = self.sess_.run([
            self.parser_.evaluation['epochs'],
            self.parser_.evaluation['eval_metrics'],
            self.parser_.evaluation['documents']
       ])

      sink_documents = tf.placeholder(tf.string)
      sink = gen_parser_ops.document_sink(sink_documents,
                                          task_context=self.task_context_,
                                          corpus_name='stdout-conll')

      self.sess_.run(sink, feed_dict={sink_documents: tf_documents})
      sys.stdout.flush()

  def _read_all_stream(self, strm):
    result = str()

    max_read = 1024
    while True:
      chunk = os.read(strm.fileno(), max_read)
      result += chunk
      if len(chunk) < max_read:
        break

    return result


class SyncHandler(SocketServer.BaseRequestHandler):
  def handle(self):
    logger.debug('Incoming request.')
    data = self._read_incoming_request()
    #data = self._read_all_from_socket(self.request)

    logger.debug('Morphological analysis...')
    morph_result = self.server.morpher_.parse(data)
    logger.debug('Done.')

    logger.debug('Tagging...')
    tagging_result = self.server.tagger_.parse(morph_result)
    logger.debug('Done.')

    logger.debug('Parsing...')
    parsing_result = self.server.parser_.parse(tagging_result)
    logger.debug('Done.')

    result = parsing_result
    self.request.sendall(result)

  def _read_incoming_request(self):
    data = str()

    while True:
      chunk = self.request.recv(1024)
      data += chunk
      if '\n\n' in data:
        break

    return data  


def configure_stdout():
  read_stream, write_stream = os.pipe()
  os.dup2(write_stream, sys.stdout.fileno())
  return os.fdopen(read_stream)


def main():
  import argparse

  parser = argparse.ArgumentParser(description = 'Syntaxnet server.') 
  parser.add_argument('--host', 
                      required = True, 
                      help = 'Accepted hosts',
                      default = '0.0.0.0')

  parser.add_argument('--port', 
    required = True, 
    help = 'Listening port',
    default = 9999)

  args = parser.parse_args()

  sync_server = SocketServer.TCPServer((args.host, int(args.port)), SyncHandler)
  stdout_strm = configure_stdout()
  sync_server.morpher_ = ProcessorSyntaxNet(CFG_MORPH_PARSER, stdout_strm)
  sync_server.tagger_ = ProcessorSyntaxNet(CFG_MORPH_TAGGER, stdout_strm)
  sync_server.parser_ = ProcessorSyntaxNet(CFG_SYNTAX_PARSER, stdout_strm)
  sync_server.serve_forever()  


if __name__ == '__main__':
  main()
