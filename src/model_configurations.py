from function_transformer_attention import ODEFuncTransformerAtt
from function_GAT_attention import ODEFuncAtt
from function_laplacian_diffusion import LaplacianODEFunc, LaplacianODEFunc1, LaplacianODEFunc2

from function_laplacian_convection import ODEFuncLapCONV

from function_GAT_convection import ODEFuncAttConv


from function_transformer_convection import ODEFuncTransConv


from block_constant_fractional import ConstantODEblock_FRAC, ConstantODEblock_FRAC1, ConstantODEblock_FRAC2, ConstantODEblock_FRAC3
from block_transformer_fractional import AttODEblock_FRAC, AttODEblock_FRAC1, AttODEblock_FRAC2, AttODEblock_FRAC3, AttODEblock_FRAC4, AttODEblock_FRAC5, AttODEblock_FRAC6, AttODEblock_FRAC7

from function_laplacian_graphcon import LaplacianODEFunc_graphcon
from function_transformer_graphcon import ODEFuncTransformerAtt_graphcon
from function_GAT_graphcon import ODEFuncAtt_graphcon

from block_attention_graph import AttODEblock_GRAPH
from block_constant_graph import ConstantODEblock_GRAPH



class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass


def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'constant_frac':

    block = ConstantODEblock_FRAC2

  elif ode_str == 'att_frac':

    block = AttODEblock_FRAC5
  elif ode_str == 'att_graph':
    block = AttODEblock_GRAPH
  elif ode_str == 'constant_graph':
    block = ConstantODEblock_GRAPH
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  elif ode_str == 'GAT':
    f = ODEFuncAtt
  elif ode_str == 'transformer':
    f = ODEFuncTransformerAtt
  elif ode_str == 'lapconv':
    f = ODEFuncLapCONV
  elif ode_str == 'gatconv':
    f = ODEFuncAttConv
  elif ode_str == 'transconv':
    f = ODEFuncTransConv
  elif ode_str == 'lapgraphcon':
    f = LaplacianODEFunc_graphcon
  elif ode_str == 'transgraphcon':
    f = ODEFuncTransformerAtt_graphcon
  elif ode_str == 'gatgraphcon':
    f = ODEFuncAtt_graphcon
  else:
    raise FunctionNotDefined
  return f
