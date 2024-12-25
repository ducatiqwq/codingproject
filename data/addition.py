from .columns import construct_colums_addition
from .templates import user_prompt, model_response

def get_data(number_1: int, number_2: int):
	numbers_sum = number_1 + number_2
	columns_addition = construct_colums_addition(number_1, number_2)

	info = {
		"number_1": number_1,
		"number_2": number_2,
		"numbers_sum": numbers_sum,
		"colums_addition": columns_addition
	}
	return user_prompt.format(**info) + '\n' + model_response.format(**info)


if __name__ == "__main__":
	data = get_data(7823, 6939)
	print(data)