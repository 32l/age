#  parse user-input options

import argparse


def parse_command():
	parser = argparse.ArgumentParser()

	parser.add_argument('command', type = str, default = 'help',
		choices = ['train', 'test', 'help'], help = 'valid commands: train, test, help')


	command = parser.parse_known_args()[0].command

	return command


def parse_opts_age_model():

	parser = argparse.ArgumentParser()

	parser.add_argument()


if __name__ == '__main__':

	# test parse_command()

	command = parse_command()

	print(command)