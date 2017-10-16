import argparse

ps_1 = argparse.ArgumentParser(add_help = False)
ps_1.add_argument('--arg1', type = int, default = 1)

ps_2 = argparse.ArgumentParser(parents = [ps_1])
ps_2.add_argument('--arg2', type = int, default = 2)


args = ps_2.parse_args()

print(args)