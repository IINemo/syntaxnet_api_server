from syntaxnet import structured_graph_builder
from syntaxnet.ops import gen_parser_ops
from syntaxnet import task_spec_pb2

import tensorflow as tf
from tensorflow.python.platform import gfile

from google.protobuf import text_format

import tempfile
import sys
import os


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
    stdout_file_path,
    flush_input = False,
    max_tmp_size = 262144000):

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
    self.stdout_file_path = stdout_file_path
    self.max_tmp_size = max_tmp_size
    self.init_line = init_line


def RewriteContext(task_context_file, resource_dir):
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
  def __init__(self, cfg):
    super(ProcessorSyntaxNet, self).__init__()

    self.cfg_ = cfg
    self.parser_ = None
    self.task_context_ = RewriteContext(self.cfg_.task_context_file,
                                        self.cfg_.resource_dir)
    self.sess_ = tf.Session()

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
    if self.cfg_.flush_input and os.stat(self.cfg_.custom_file_path).st_size > self.cfg_.max_tmp_size:
      logger.debug('Cleaning input file.')
      with open(self.cfg_.custom_file_path, 'w') as f:
        pass
      logger.debug('Done.') 

      logger.debug('Reseting offset inside tensorflow input file class.')
      self._parse_impl()
      logger.debug('Done.')
    
    with open(self.cfg_.custom_file_path, 'a') as f:
      f.write(raw_bytes)
      f.flush()

    self._parse_impl()

    result = self._read_all_stream()
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

      sys.stdout.write('\n')
      sys.stdout.flush()

  def _read_all_stream(self):
    with open(self.cfg_.stdout_file_path, 'r') as f:
      result = f.read()
    
    os.ftruncate(sys.stdout.fileno(), 0)
    os.lseek(sys.stdout.fileno(), 0, 0)
    sys.stdout.flush()
    
    result = result[:-1]
    return result
