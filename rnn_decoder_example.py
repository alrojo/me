class rnn_decoder():
  def __init__(self, cell, initial_state, decoder_fn, inputs=None,
               sequence_length=None, output_fn=None, state_fn=None,
               loop_fn=None, encoder_projection=None, parallel_iterations=None,
               swap_memory=False, time_major=False, scope=None):
    self.cell = cell
    self.initial_state = initial_state
    self.decoder_fn = decoder_fn(self)
    self.inputs = inputs
    self.sequence_length = sequence_length
    self.encoder_projection = encoder_projection
    self.parallel_iterations = parallel_iterations
    self.swap_memory = swap_memory
    self.time_major = time_major
    self.scope = scope

  def run(self):
    if self.encoder_projection is None:
      # Project initial_state as described in Bahdanau et al. 2014
      # https://arxiv.org/abs/1409.0473
      self.state = layers.fully_connected(self.initial_state, self.cell.output_size,
                                   activation_fn=math_ops.tanh)
    else:
      self.state = self.initial_state
    # Testing input dimensions
    if len(self.inputs.get_shape()) is not 3:
      raise ValueError("Inputs must have three dimensions")
    if self.inputs.get_shape()[2] is None:
      raise ValueError("Inputs must not be `None` in the feature (3'rd) "
                       "dimension")
    # Setup of RNN (dimensions, sizes, length, initial state, dtype)
    # Setup dtype
    self.dtype = self.state.dtype
    if not self.time_major:
      # [batch, seq, features] -> [seq, batch, features]
      self.inputs = array_ops.transpose(self.inputs, perm=[1, 0, 2])
    # Get data input information
    self.batch_size = array_ops.shape(self.inputs)[1]
    self.input_depth = int(self.inputs.get_shape()[2])
    # Setup decoder inputs as TensorArray
    self.inputs_ta = tensor_array_ops.TensorArray(self.dtype, size=0, dynamic_size=True)
    self.inputs_ta = self.inputs_ta.unpack(self.inputs)

    # Run raw_rnn function
    outputs_ta, _, _ = (
        rnn.raw_rnn(self.cell, self.decoder_fn,
                    parallel_iterations=self.parallel_iterations,
                    swap_memory=self.swap_memory, scope=varscope))
    outputs = outputs_ta.pack()
    if not self.time_major:
      # [seq, batch, features] -> [batch, seq, features]
      outputs = array_ops.transpose(outputs, perm=[1, 0, 2])
    return outputs
    
# resetting the graph
reset_default_graph()

# Setting up hyperparameters and general configs
MAX_DIGITS = 5
MIN_DIGITS = 5
NUM_INPUTS = 27
NUM_OUTPUTS = 11 #(0-9 + '#')

BATCH_SIZE = 100
# try various learning rates 1e-2 to 1e-5
LEARNING_RATE = 0.005
X_EMBEDDINGS = 8
t_EMBEDDINGS = 8
NUM_UNITS_ENC = 10
NUM_UNITS_DEC = 10


# Setting up placeholders, these are the tensors that we "feed" to our network
Xs = tf.placeholder(tf.int32, shape=[None, None], name='X_input')
ts_in = tf.placeholder(tf.int32, shape=[None, None], name='t_input_in')
ts_out = tf.placeholder(tf.int32, shape=[None, None], name='t_input_out')
X_len = tf.placeholder(tf.int32, shape=[None], name='X_len')
t_len = tf.placeholder(tf.int32, shape=[None], name='X_len')
t_mask = tf.placeholder(tf.float32, shape=[None, None], name='t_mask')

# Building the model

# first we build the embeddings to make our characters into dense, trainable vectors
X_embeddings = tf.get_variable('X_embeddings', [NUM_INPUTS, X_EMBEDDINGS],
                               initializer=tf.random_normal_initializer(stddev=0.1))
t_embeddings = tf.get_variable('t_embeddings', [NUM_OUTPUTS, t_EMBEDDINGS],
                               initializer=tf.random_normal_initializer(stddev=0.1))

# setting up weights for computing the final output
W_out = tf.get_variable('W_out', [NUM_UNITS_DEC, NUM_OUTPUTS])
b_out = tf.get_variable('b_out', [NUM_OUTPUTS])

output_projection = [W_out, b_out]

X_embedded = tf.gather(X_embeddings, Xs, name='embed_X')
t_embedded = tf.gather(t_embeddings, ts_in, name='embed_t')

# forward encoding
enc_cell = tf.nn.rnn_cell.GRUCell(NUM_UNITS_ENC)#python.ops.rnn_cell.GRUCell
_, enc_state = rnn.dynamic_rnn(cell=enc_cell, inputs=X_embedded, dtype=tf.float32,
                               sequence_length=X_len)
# use below incase TF's makes issues
#enc_state, _ = tf_utils.encoder(X_embedded, X_len, 'encoder', NUM_UNITS_ENC)
#
#enc_state = tf.concat(1, [enc_state, enc_state])

# decoding
# note that we are using a wrapper for decoding here, this wrapper is hardcoded to only use GRU
# check out tf_utils to see how you make your own decoder

dec_cell = tf.nn.rnn_cell.GRUCell(NUM_UNITS_DEC)

def decoder_fn_wrapper(inputs_fn):
  def decoder_fn(self):
    def loop_fn(time, cell_output, cell_state, loop_state):
      elements_finished = (time >= self.sequence_length) #TODO handle seq_len=None
      # get s_t, y_t
      emit_output = cell_output
      if cell_output is None:
        next_cell_state = self.state
      else:
        next_cell_state = cell_state
      # get x_{t+1}
      next_input, elements_finished = inputs_fn(self, time, next_cell_state, elements_finished)
      # get loop_state
      next_loop_state = loop_state
      return (elements_finished, next_input, next_cell_state,
              emit_output, next_loop_state)
    return loop_fn
  return decoder_fn


def inputs_fn_train(self, time, state, elements_finished):
  finished = math_ops.reduce_all(elements_finished)
  return control_flow_ops.cond(
      finished,
      lambda: array_ops.zeros([self.batch_size, self.input_depth], dtype=self.dtype),
      lambda: self.inputs_ta.read(time)), elements_finished

def inputs_fn_eval_wrapper(embeddings, eos_symbol, output_fn):
  def inputs_fn_eval(self, time, state, elements_finished):
    finished = math_ops.reduce_all(elements_finished)
    next_input = control_flow_ops.cond(
        finished,
        # zero state handling
        lambda: array_ops.zeros([self.batch_size, self.input_depth],
                                dtype=self.dtype),
        lambda: control_flow_ops.cond(math_ops.greater(time, 0),
            # get embedding
            lambda: tf.gather(embeddings, tf.argmax(output_fn(self, state), 1)), # Gather max prediction.
            # read at eos-tag
            lambda: self.inputs_ta.read(0))) # Read <EOS> tag
    print next_input.get_shape()
    return next_input, elements_finished
  return inputs_fn_eval

def cool_inputs_fn_eval_wrapper(embeddings, eos):
    def inputs_fn_eval(self, time, state, elements_finished):
      next_input_id = control_flow_ops.cond(math_ops.greater(time, 0),
          lambda: tf.argmax(self.output_fn(self, state), 1),
          lambda: tf.ones([self.batch_size], dtype=tf.int64) * eos)
      # fetching embedding
      embedding = tf.gather(embeddings, next_input_id)
      # if time != 0, check if embedding is eos and update elements_finished
      elements_finished = control_flow_ops.cond(math_ops.greater(time, 0),
          lambda: math_ops.logical_or(elements_finished,
                                      next_input_id == eos),
          lambda: elements_finished)
      print embedding.get_shape()
      return embedding, elements_finished
    return inputs_fn_eval


print(t_embedded)
with vs.variable_scope("decoding") as varscope:
  output_fn = lambda self, x: tf.matmul(x, W_out) + b_out
  dec_out = rnn_decoder(cell=dec_cell,
                        decoder_fn=decoder_fn_wrapper(inputs_fn_train),
                        inputs=t_embedded,
                        initial_state=enc_state,
                        sequence_length=t_len).run()
  varscope.reuse_variables()
  inputs_fn_eval = inputs_fn_eval_wrapper(t_embeddings, data_generator.eos_symbol, output_fn)
  valid_dec_out = rnn_decoder(cell=dec_cell,
                        decoder_fn=decoder_fn_wrapper(inputs_fn_eval),
                        inputs=t_embedded,
                        initial_state=enc_state,
                        output_fn=output_fn,
                        sequence_length=t_len).run()
  # reshaping to have [batch_size*seqlen, num_units]
  out_tensor = tf.reshape(dec_out, [-1, NUM_UNITS_DEC])
  valid_out_tensor = tf.reshape(valid_dec_out, [-1, NUM_UNITS_DEC])
  # computing output
  out_tensor = output_fn(None, out_tensor)
  valid_out_tensor = output_fn(None, valid_out_tensor)
  # reshaping back to sequence
  b_size = tf.shape(X_len)[0] # use a variable we know has batch_size in [0]
  seq_len = tf.shape(t_embedded)[1] # variable we know has sequence length in [1]
  num_out = tf.constant(NUM_OUTPUTS) # casting NUM_OUTPUTS to a tensor variable
  out_shape = tf.concat(0, [tf.expand_dims(b_size, 0),
                            tf.expand_dims(seq_len, 0),
                            tf.expand_dims(num_out, 0)])
  out_tensor = tf.reshape(out_tensor, out_shape)
  valid_out_tensor = tf.reshape(valid_out_tensor, out_shape)
  # handling shape loss
  #out_tensor.set_shape([None, None, NUM_OUTPUTS])
  y = out_tensor
  y_valid = valid_out_tensor
