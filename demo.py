import argparse

parser = argparse.ArgumentParser(description="RL Toolkit")
parser.add_argument(
    "-a",
    "--algorithm",
    type=str,
    choices=["dqn", "q_learning", "ddpg"],
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

if args.algorithm == "dqn":
    import dqn.tester
    import dqn.trainer

    if args.mode == "train":
        trainer = dqn.trainer.Trainer()
        trainer.train()
    elif args.mode == "test":
        tester = dqn.tester.Tester()
        tester.test()
elif args.algorithm == "q_learning":
    import q_learning.tester
    import q_learning.trainer

    if args.mode == "train":
        trainer = q_learning.trainer.Trainer()
        trainer.train()
    elif args.mode == "test":
        tester = q_learning.tester.Tester()
        tester.test()
elif args.algorithm == "ddpg":
    import ddpg.tester
    import ddpg.trainer

    if args.mode == "train":
        trainer = ddpg.trainer.Trainer()
        trainer.train()
    elif args.mode == "test":
        tester = ddpg.tester.Tester()
        tester.test()
