# 代码说明

display_...为展示训练后的模型效果使用的文件

train							—— Dueling DDQN训练使用
train_a3c					——a3c不能使用，没调bug
train_actor_critic	 ——actor critic 没训练出来好的结果
train_PPO				  ——PPO训练使用
icm_gae_ppo			——gae+PPO训练使用
rainbow					——rainbow训练使用

由于Replay文件夹太大，我们给删去了

model文件夹存model，param文件夹存的param， info文件夹里有训练记录的奖励,loss，和血量差信息