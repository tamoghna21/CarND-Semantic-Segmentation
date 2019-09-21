import os.path
import tensorflow as tf

loaded_graph = tf.Graph()
load_dir = './model.ckpt'

with tf.Session(graph=loaded_graph) as sess:
	#Load saved model
	loader = tf.train.import_meta_graph(load_dir + '.meta')
	loader.restore(sess, load_dir)
    
	#tf.get_default_graph().as_graph_def()
	graph_def = loaded_graph.as_graph_def() #current Graph as protobuffs
    
	# Parameters: 1) graph, 2) directory where we want to save the pb file,
	#             3) name of the file, 4) text format (True) or binary format.
	tf.train.write_graph(graph_def,".","graph.pb",False)
    

print("Done")