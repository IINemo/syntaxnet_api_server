# -*- coding: utf8 -*-

################################################################################
# Configuring the PATH

import sys
import os


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
# Configuring logging

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
# Hack: coping the proper configuration 

import shutil

custom_file_path = '/root/models/syntaxnet/bazel-bin/syntaxnet/parser_eval.runfiles/__main__/syntaxnet/api/context.pbtxt'
orig_file_path = '/root/models/syntaxnet/syntaxnet/models/parsey_universal/context.pbtxt'
shutil.copyfile(custom_file_path, orig_file_path)

################################################################################
# Configurations for SyntaxNet processors

from processor_syntaxnet import ProcessorSyntaxNetConfig


task_context_file = '/root/models/syntaxnet/syntaxnet/models/parsey_universal/context.pbtxt'
resource_dir = '/root/models/syntaxnet/syntaxnet/models/Model1'
custom_file_dir = '/dev/shm/'
stdout_file_path =  '/dev/shm/stdout.tmp'


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
  stdout_file_path = stdout_file_path,
  task_context_file = task_context_file,
  flush_input = True,
  max_tmp_size = 262144000,
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
  flush_input = True,

  custom_file_path = os.path.join(custom_file_dir, 'tagger.tmp'),
  input_str = 'custom_file_tagger',
  stdout_file_path = stdout_file_path,
  task_context_file = task_context_file,
  variable_scope = 'tagger',
  max_tmp_size = 262144000,
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
  flush_input = True,

  custom_file_path = os.path.join(custom_file_dir, 'parser.tmp'),
  input_str = 'custom_file_parser',
  stdout_file_path = stdout_file_path,
  task_context_file = task_context_file,
  variable_scope = 'synt_parser',
  max_tmp_size = 262144000,
  init_line = '1\t'*12)

################################################################################

import SocketServer
from processor_syntaxnet import ProcessorSyntaxNet


class SyncHandler(SocketServer.BaseRequestHandler):
  def handle(self):
    logger.debug('Incoming request.')
    data = self._read_incoming_request()

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
    
    while not result.endswith('\n\n\n'):
      result += '\n'

    self.request.sendall(result)

  def _read_incoming_request(self):
    data = str()

    while True:
      chunk = self.request.recv(51200)
      data += chunk
      if '\n\n' in data:
        break

    return data  


def configure_stdout(stdout_file_path):
  strm = open(stdout_file_path, 'w') # bypassing linux 64 kb pipe limit
  os.dup2(strm.fileno(), sys.stdout.fileno())
  return strm


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
  stdout_strm = configure_stdout(stdout_file_path)
  sync_server.morpher_ = ProcessorSyntaxNet(CFG_MORPH_PARSER)
  sync_server.tagger_ = ProcessorSyntaxNet(CFG_MORPH_TAGGER)
  sync_server.parser_ = ProcessorSyntaxNet(CFG_SYNTAX_PARSER)
  sync_server.serve_forever()  


if __name__ == '__main__':
  main()
