import q_learning
import q_learning.tester
import q_learning.trainer

# q_learning
trainer = q_learning.trainer.Trainer()
tester = q_learning.tester.Tester()
trainer.train()
tester.test()
