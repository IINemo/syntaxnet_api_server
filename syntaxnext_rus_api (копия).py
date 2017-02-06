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


beam_size = 8
max_steps = 1000
arg_prefix = 'brain_morpher'
slim_model = True
task_context_file = '/root/models/syntaxnet/syntaxnet/models/parsey_universal/context.pbtxt'
resource_dir = '/root/models/syntaxnet/syntaxnet/models/Russian-SynTagRus'
model_path = os.path.join(resource_dir, 'morpher-params')
#input_str = 'stdin-conll'
#input_str = 'stdin'

#input_str = '/root/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/__main__/syntaxnet/api/1.txt'
input_str = 'custom_file'
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
  print parser.evaluation.keys()
  sys.stdout.flush()
  
  parser.AddSaver(slim_model)
  sess.run(parser.inits.values())
  parser.saver.restore(sess, model_path)

  
  return parser
  #return parser
  

def parse(parser, task_context):
  
  # parser.AddEvaluation(task_context,
  #                      batch_size,
  #                      corpus_name=input_str,
  #                      evaluation_max_steps=max_steps)
  
  #sess.run(parser.inits.values())
  #parser.saver.restore(sess, model_path)


  print '~~~~~~', parser.evaluation['documents']
  
  print parser.evaluation
  sys.stdout.flush()
  # parser.evaluation.update(parser._AddBeamReader(task_context,
  #                                  batch_size,
  #                                  'stdin',
                                   # until_all_final=True,
                                   # always_start_new_sentences=True))
  tf_eval_epochs, tf_eval_metrics, tf_documents = sess.run([
        parser.evaluation['epochs'],
        parser.evaluation['eval_metrics'],
        parser.evaluation['documents']
   ])

  print tf_documents

  sink_documents = tf.placeholder(tf.string)
  sink = gen_parser_ops.document_sink(sink_documents,
                                      task_context=task_context,
                                      corpus_name=output_str)

  sess.run(sink, feed_dict={sink_documents: tf_documents})

custom_file_path = '/root/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/__main__/syntaxnet/api/1.txt'

class SyncHandler(SocketServer.BaseRequestHandler):
  def handle(self):
    data = self.request.recv(1024)
    print 'RRR', data
    with open(custom_file_path, 'w') as f:
      f.write(data)
    #self.make_child_process(data)
    self.pipe_trick(data)

  def process_child(self):
    print 'WEEEEE'
    sys.stdout.flush()
    #data = sys.stdin.read()
    #print data
    parse(self.server.parser_, self.server.task_context_)
    print '222222'
    sys.stdout.flush()
    #sys.stdout.close()
    #exit(0)

  def process_parent(self, data, pipe_write, pipe_read):
    pipe_write.write(data)
    pipe_write.write("\n\n") 
    pipe_write.flush()
    #pipe_write.close()
    #os.close(pipe_write.fileno())
    #pipe_write.close()

    #del pipe_write
    print '7777777777777777777'
    sys.stdout.flush()

    reply = pipe_read.read()

    #reply = pipe_read.read(-1)
    print '9999999999999999999999'
    print reply
    #print reply
    sys.stdout.flush()
    time.sleep(10)

  def make_child_process(self, data):
    stdin_slave, stdout_master = os.pipe()
    stdin_master, stdout_slave  = os.pipe()

    pid = os.fork()
    if pid == 0:
        os.dup2(stdin_slave, sys.stdin.fileno())
        os.dup2(stdout_slave, sys.stdout.fileno())

        os.close(stdin_slave)
        os.close(stdout_slave)
        os.close(stdout_master)
        os.close(stdin_master)

        self.process_child()

    else:
        os.close(stdin_slave)
        os.close(stdout_slave)

        pipe_write = os.fdopen(stdout_master, 'w')
        pipe_read = os.fdopen(stdin_master, 'r')
        self.process_parent(data, pipe_write, pipe_read)

  def pipe_trick(self, data):
    r, w = os.pipe()
    os.dup2(r, sys.stdin.fileno())

    write_stream = os.fdopen(w, 'w')
    
    print type(data)
    #os.write(w, data)
    write_stream.write(data)
    write_stream.flush()
    write_stream.close()
    #os.flush(w)
    #os.fsync(w)
    #os.close(w)

    parse(self.server.parser_, self.server.task_context_)
    os.close(r)
    print '??????????????'
    sys.stdout.flush()

  def pipe_fork_trick(self, data):
    # pid = os.fork()
    # if pid == 0:
    r, w = os.pipe()
    os.dup2(r, sys.stdin.fileno())
    write_stream = os.fdopen(w, 'w')
    write_stream.write(data)
    write_stream.flush()
    write_stream.close()
    parse(self.server.parser_, self.server.task_context_)
    os.close(r)
    # else:
    #   time.sleep(7)

def main(sess):
  sync_server = SocketServer.TCPServer(('0.0.0.0', 9999), SyncHandler)
  sync_server.task_context_ = RewriteContext(task_context_file)
  sync_server.parser_ = init(sess, sync_server.task_context_)
  sync_server.serve_forever()  


if __name__ == '__main__':
  with tf.Session() as sess:
    main(sess)
  
