# 第1步：命令行参数解析，获取集群的信息ps_hosts和worker_hosts，以及当前节点的角色信息job_name和task_index。例如:
tf.app.flags.DEFINE_string("ps_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "", "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
FLAGS = tf.app.flags.FLAGS
ps_hosts = FLAGS.ps_hosts.split(",")
worker_hosts = FLAGS.worker_hosts(",")

# 第2步：创建当前任务节点的服务器
cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)

# 第3步：如果当前节点是参数服务器，则调用server.join()无休止等待；如果是工作节点，则执行第4步
if FLAGS.job_name == "ps":
  server.join()
# 第4步：构建要训练的模型，构建计算图
elif FLAGS.job_name == "worker":
# build tensorflow graph model

# 第5步：创建tf.train.Supervisor来管理模型的训练过程
# 创建一个supervisor来监督训练过程
sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0), logdir="/tmp/train_logs")
# supervisor负责会话初始化和从检查点恢复模型
sess = sv.prepare_or_wait_for_session(server.target)
# 开始循环，直到supervisor停止
while not sv.should_stop()
   # 训练模型