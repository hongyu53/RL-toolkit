import argparse

parser = argparse.ArgumentParser(description="RL Toolkit")
parser.add_argument(
    "-a",
    "--algorithm",
    type=str,
    choices=["q_learning", "dqn", "ddqn", "d3qn", "ddpg"],
    help="select algorithm",
)
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    choices=["train", "test"],
    help="select mode to train or test",
)
args = parser.parse_args()

if args.algorithm == "q_learning":
    if args.mode == "train":
        from q_learning.trainer import Trainer

        trainer = Trainer()
        trainer.train()
    elif args.mode == "test":
        from q_learning.tester import Tester

        tester = Tester()
        tester.test()
elif args.algorithm == "dqn":
    if args.mode == "train":
        from dqn.trainer import Trainer

        trainer = Trainer()
        trainer.train()
    elif args.mode == "test":
        from dqn.tester import Tester

        tester = Tester()
        tester.test()
elif args.algorithm == "ddqn":
    if args.mode == "train":
        from ddqn.trainer import Trainer

        trainer = Trainer()
        trainer.train()
    elif args.mode == "test":
        from ddqn.tester import Tester

        tester = Tester()
        tester.test()
elif args.algorithm == "d3qn":
    if args.mode == "train":
        from d3qn.trainer import Trainer

        trainer = Trainer()
        trainer.train()
    elif args.mode == "test":
        from d3qn.tester import Tester

        tester = Tester()
        tester.test()
elif args.algorithm == "ddpg":
    if args.mode == "train":
        from ddpg.trainer import Trainer

        trainer = Trainer()
        trainer.train()
    elif args.mode == "test":
        from ddpg.tester import Tester

        tester = Tester()
        tester.test()
