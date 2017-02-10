import sys
task_number = int(sys.argv[1])
job_name = str(sys.argv[2])

import tensorflow as tf 


cluster = tf.train.ClusterSpec({"local" : ["localhost:2222"],"ps" : ["localhost:2223", "localhost:2224"]})
server = tf.train.Server(cluster, job_name = job_name, task_index = task_number)

print("Starting server #{}".format(task_number))

server.start()
server.join()
