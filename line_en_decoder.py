# coding=utf-8
import sys


def encoder_line(line):
	points = ['.', '..', '...', ',', '!', '!!!', '~']
	encoder_line = ''
	for word in line.split(' '):
		word_number = ''
		for char in word:
			if char in points:
				word_number += char
			else:
				word_number += str(ord(char))+'_'
		encoder_line += word_number
		encoder_line += '_'
	encoder_line = encoder_line[:-1]
	return encoder_line 


def decoder_line(encoder_):
	points = ['.', '..', '...', ',', '!', '!!!', '~']
	encoder_line = ''
	for number in encoder_.split('__'):
		word_ = ''
		for num_ in number.split('_'):
			if num_ in points:
				word_ += num_
			else:
				word_ += chr(int(num_))
		word_ += ' '
		encoder_line += word_

	return encoder_line 

if __name__ == "__main__":

	line = sys.argv[1]
	encoder_ = encoder_line(line)
	print('加密结果: ', encoder_)
	line_ = decoder_line(encoder_)
	print('解码结果: ', line_)

